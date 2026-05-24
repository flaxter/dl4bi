"""Open-weights LLM scorer for FlowGP black-box conditioning.

FlowGP's gradient-free guidance (Fisher identity, paper Eq. 35) only requires
forward *evaluations* of log q(C | f0): the per-step weights w^(i) are a softmax
over scalar scores of S candidate trajectories drawn from p(f0 | f_t). This
module supplies q(C | f0) by an open-weights instruction-tuned LLM:

    (1) yes/no verifier ('yesno')
        Prompt:  '<description>\n\n<trajectory rendered as text>\n\n
                  Is this function consistent with the description? Answer Yes or No.'
        Score:   log P(Yes | prompt) - log P(No | prompt)
        Cheap (1 forward pass) and well-calibrated as a relative score.

    (2) sequence log-probability ('seqlp', a la Biggs & Willis)
        Condition on '<description>\n\nThe function values are:\n'
        and return sum_t log P(value_token_t | preceding context).
        The principled density q(C | f0) (= p(values | C) under the LLM).

Both modes are *batched* over the (N*S, M) candidates produced per ODE step:
one call to model.forward over the full batch of rendered prompts.

The class `LLMVerifier` wraps a HuggingFace causal LM; `MockVerifier` uses the
existing programmatic verifier so the integration can be smoke-tested without
a GPU. `make_score_fn` returns a callable suitable for the FlowGP guidance
loop: it accepts a JAX array of shape (..., M), reshapes to (B, M), calls the
underlying scorer, and returns a JAX array of shape (...).

The fusion samplers `blackbox_guided_sample` and `fuse` use this callable from
inside a Python for-loop variant of the ODE integrator (`*_pyloop` in the
respective scripts) -- LLM calls are not jittable.
"""

from __future__ import annotations

from typing import Callable, Sequence

import jax
import jax.numpy as jnp
import numpy as np


# ---------- trajectory -> text -------------------------------------------------


def render_trajectory(
    f: np.ndarray,
    x: np.ndarray,
    n_show: int = 12,
    digits: int = 2,
    x_digits: int = 2,
) -> str:
    """Render f(x) as a compact list of values for LLM consumption.

    Subsamples to n_show evenly-spaced points so the prompt stays short
    (typical M=64-80, n_show=12 keeps the answer-conditioning context tight).
    """
    M = f.shape[-1]
    if n_show >= M:
        idx = np.arange(M)
    else:
        idx = np.linspace(0, M - 1, n_show).round().astype(int)
    parts = []
    for i in idx:
        parts.append(f"x={x[i]:.{x_digits}f}: {f[i]:+.{digits}f}")
    return "\n".join(parts)


# ---------- prompt construction -----------------------------------------------


def build_yesno_prompt(description: str, rendered: str) -> tuple[str, str]:
    """Return (system, user) text for a chat-template yes/no verifier."""
    system = (
        "You are a careful evaluator of whether a numerical function matches a "
        "natural-language description. You answer only 'Yes' or 'No'."
    )
    user = (
        f"Description: {description}\n\n"
        f"Function values f(x) on the grid x in [0,1]:\n{rendered}\n\n"
        "Question: Is this function consistent with the description? "
        "Answer with a single token: Yes or No."
    )
    return system, user


def build_seqlp_context(description: str) -> str:
    """Prefix conditioning text for the sequence-logprob scorer. The continuation
    is the rendered trajectory; we score sum log p(continuation | prefix)."""
    return (
        f"Description: {description}\n\n"
        "Function values f(x) on the grid x in [0,1]:\n"
    )


# ---------- backends -----------------------------------------------------------


class MockVerifier:
    """Calls a user-supplied JAX function as a stand-in for an LLM.

    The sampler interface is unchanged, so this is the right way to smoke-test
    plumbing on machines without torch/GPU: pass the existing programmatic
    verifier (e.g. ``describe_logscore``, ``sitrep_score``) and you get back
    the same numbers, with the same shape contract, that an LLM would produce.
    """

    def __init__(self, score_fn: Callable[[jnp.ndarray], jnp.ndarray]):
        self._fn = score_fn
        self.calls = 0
        self.total_prompts = 0

    def score(self, f_batch: np.ndarray, description: str) -> np.ndarray:
        del description  # mock ignores text
        self.calls += 1
        self.total_prompts += f_batch.shape[0]
        out = np.asarray(self._fn(jnp.asarray(f_batch)))
        return out.astype(np.float64)

    def close(self):  # pragma: no cover - mock has nothing to release
        pass


class LLMVerifier:
    """Open-weights instruction-tuned LM as q(C | f0) scorer.

    `score(f_batch, description)` returns a (B,) numpy array of log-scores.
    Modes:
        'yesno' -- log P(Yes|prompt) - log P(No|prompt) at the first generated
                   answer token (1 forward pass per prompt). Default.
        'seqlp' -- summed token log-prob of the rendered trajectory tokens
                   given a description-conditioned prefix.

    The model is loaded lazily on first call; tokenizer applies the chat
    template if available. Batched via left-padding so all rows have a common
    final-position index.

    Parameters
    ----------
    model_name : HF model id, e.g. 'Qwen/Qwen2.5-1.5B-Instruct'.
    device     : 'cuda' / 'cpu' / 'auto'.
    dtype      : torch dtype, default bfloat16 on GPU else float32.
    mode       : 'yesno' or 'seqlp'.
    x_grid     : numpy array of x positions used by `render_trajectory`.
    n_show     : how many grid points to render in the prompt.
    digits     : decimal digits per value.
    max_batch  : split incoming f_batch into chunks of at most this many rows
                 (default 64) to bound peak KV-cache usage.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        device: str = "auto",
        dtype: str | None = None,
        mode: str = "yesno",
        x_grid: np.ndarray | None = None,
        n_show: int = 12,
        digits: int = 2,
        max_batch: int = 64,
    ):
        if mode not in {"yesno", "seqlp"}:
            raise ValueError(f"unknown mode {mode!r}, expected 'yesno' or 'seqlp'")
        if x_grid is None:
            raise ValueError("x_grid is required to render trajectories")
        self.model_name = model_name
        self.mode = mode
        self.x_grid = np.asarray(x_grid)
        self.n_show = n_show
        self.digits = digits
        self.max_batch = max_batch
        self.calls = 0
        self.total_prompts = 0

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if dtype is None:
            torch_dtype = torch.bfloat16 if device == "cuda" else torch.float32
        else:
            torch_dtype = getattr(torch, dtype)
        self.device = device

        print(f"[LLMVerifier] loading {model_name} on {device} ({torch_dtype})")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=torch_dtype
        ).to(device)
        self.model.eval()

        # Yes/No token ids. Use the variant the chat template will continue with.
        # Most chat templates put the assistant prompt right before generation;
        # we score the FIRST token of "Yes" / "No".
        self._yes_id = self._first_token_id(" Yes") or self._first_token_id("Yes")
        self._no_id = self._first_token_id(" No") or self._first_token_id("No")
        if self._yes_id is None or self._no_id is None:
            raise RuntimeError("could not resolve Yes/No token ids")

    def _first_token_id(self, s: str) -> int | None:
        toks = self.tokenizer.encode(s, add_special_tokens=False)
        return toks[0] if toks else None

    # ----- yesno --------------------------------------------------------------

    def _build_yesno_prompts(
        self, f_batch: np.ndarray, description: str
    ) -> list[str]:
        prompts: list[str] = []
        for f in f_batch:
            system, user = build_yesno_prompt(
                description,
                render_trajectory(f, self.x_grid, self.n_show, self.digits),
            )
            chat = self.tokenizer.apply_chat_template(
                [{"role": "system", "content": system}, {"role": "user", "content": user}],
                tokenize=False,
                add_generation_prompt=True,
            )
            prompts.append(chat)
        return prompts

    def _score_yesno_chunk(self, prompts: list[str]) -> np.ndarray:
        torch = self.torch
        enc = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            out = self.model(**enc, use_cache=False)
        # Slice to the last (next-token) position BEFORE the float upcast --
        # materialising log_softmax over (B, L, V) is what blew memory on the
        # 152k-vocab Qwen lm_head.
        last_logits = out.logits[:, -1, :].float().contiguous()
        del out
        log_probs = torch.log_softmax(last_logits, dim=-1)
        lp_yes = log_probs[:, self._yes_id]
        lp_no = log_probs[:, self._no_id]
        return (lp_yes - lp_no).cpu().numpy().astype(np.float64)

    # ----- seqlp --------------------------------------------------------------

    def _score_seqlp_chunk(
        self, f_batch: np.ndarray, description: str
    ) -> np.ndarray:
        """Sum of log p(continuation tokens | prefix) under the LM.

        Tokenises prefix and continuation separately so we know the exact
        boundary (the tokenizer may merge bytes across the join, but we want
        to score the continuation as written, so the explicit join is safer
        than counting through a single batch encoding).
        """
        torch = self.torch
        prefix = build_seqlp_context(description)
        prefix_ids = self.tokenizer(
            prefix, add_special_tokens=True, return_tensors=None
        )["input_ids"]
        n_prefix = len(prefix_ids)

        rows_ids: list[list[int]] = []
        cont_lens: list[int] = []
        for f in f_batch:
            r = render_trajectory(f, self.x_grid, self.n_show, self.digits)
            cont_ids = self.tokenizer(r, add_special_tokens=False)["input_ids"]
            rows_ids.append(list(prefix_ids) + list(cont_ids))
            cont_lens.append(len(cont_ids))

        max_len = max(len(r) for r in rows_ids)
        pad_id = self.tokenizer.pad_token_id
        input_ids = np.full((len(rows_ids), max_len), pad_id, dtype=np.int64)
        attn = np.zeros_like(input_ids)
        for i, ids in enumerate(rows_ids):
            # left-pad
            input_ids[i, max_len - len(ids) :] = ids
            attn[i, max_len - len(ids) :] = 1

        input_ids_t = torch.from_numpy(input_ids).to(self.device)
        attn_t = torch.from_numpy(attn).to(self.device)
        with torch.no_grad():
            out = self.model(
                input_ids=input_ids_t, attention_mask=attn_t, use_cache=False
            )
        logits = out.logits  # (B, L, V) in model dtype
        del out

        # Chunked over the sequence dim to bound peak memory: float upcast +
        # logsumexp on (B, L, V) materialises the full tensor; we only need
        # (gathered - logsumexp) which we accumulate chunk by chunk.
        shifted = input_ids_t[:, 1:]
        B, Lm1 = shifted.shape
        chunk = 64
        tok_lp = torch.empty(B, Lm1, dtype=torch.float32, device=self.device)
        for s in range(0, Lm1, chunk):
            e = min(s + chunk, Lm1)
            lg = logits[:, s:e, :].float()
            tg = shifted[:, s:e]
            gathered = lg.gather(-1, tg.unsqueeze(-1)).squeeze(-1)
            lse = torch.logsumexp(lg, dim=-1)
            tok_lp[:, s:e] = gathered - lse
            del lg, gathered, lse
        del logits

        # For each row, sum tok_lp over the *last* cont_len columns of tok_lp,
        # which correspond exactly to the continuation tokens (left-padded).
        seq_lp = np.zeros(len(rows_ids), dtype=np.float64)
        tok_lp_np = tok_lp.float().cpu().numpy().astype(np.float64)
        for i, cl in enumerate(cont_lens):
            # tok_lp has length L-1; continuation occupies the last cl entries.
            if cl > 0:
                seq_lp[i] = tok_lp_np[i, -cl:].sum()
        return seq_lp

    # ----- public -------------------------------------------------------------

    def score(self, f_batch: np.ndarray, description: str) -> np.ndarray:
        self.calls += 1
        self.total_prompts += f_batch.shape[0]
        outs: list[np.ndarray] = []
        for start in range(0, f_batch.shape[0], self.max_batch):
            chunk = f_batch[start : start + self.max_batch]
            if self.mode == "yesno":
                prompts = self._build_yesno_prompts(chunk, description)
                outs.append(self._score_yesno_chunk(prompts))
            else:
                outs.append(self._score_seqlp_chunk(chunk, description))
        return np.concatenate(outs, axis=0)

    def close(self):
        try:
            del self.model
        except AttributeError:
            pass
        if self.device == "cuda":
            self.torch.cuda.empty_cache()


# ---------- JAX-compatible callable -------------------------------------------


def make_score_fn(
    backend,
    description: str | Sequence[str],
    temperature: float = 1.0,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Wrap `backend.score` as a function `f0 -> log-score` with JAX shapes.

    Accepts f0 of shape (..., M); flattens leading dims to (B, M), calls the
    backend once, reshapes back, and divides by `temperature` (so values <1
    sharpen, values >1 temper -- recall the gotcha about weight collapse).
    If `description` is a string, the same C is used for the whole batch.
    """
    if not isinstance(description, str):
        raise NotImplementedError("per-row descriptions not yet supported")
    inv_temp = 1.0 / float(temperature)

    def _fn(f0):
        f0_arr = np.asarray(f0)
        lead = f0_arr.shape[:-1]
        M = f0_arr.shape[-1]
        flat = f0_arr.reshape(-1, M)
        scores = backend.score(flat, description) * inv_temp
        return jnp.asarray(scores.reshape(lead))

    return _fn


# ---------- a tiny helper for "is the LLM seeing reasonable text" -----------


def debug_show_prompt(
    backend, f: np.ndarray, description: str, n_show: int = 12
) -> str:
    """Return the actual prompt the LLM will see for one trajectory."""
    rendered = render_trajectory(
        f, getattr(backend, "x_grid", np.linspace(0, 1, len(f))), n_show
    )
    if isinstance(backend, LLMVerifier) and backend.mode == "yesno":
        system, user = build_yesno_prompt(description, rendered)
        return f"[SYSTEM]\n{system}\n\n[USER]\n{user}"
    return f"[CONTEXT]\n{build_seqlp_context(description)}{rendered}"


# ---------- Python-loop FlowGP driver (LLM-friendly) --------------------------


def flowgp_pyloop(
    key,
    m_pred,
    K_pred,
    score_fn: Callable[[jnp.ndarray], jnp.ndarray],
    n_samples: int = 200,
    T: int = 300,
    S: int = 64,
    v_max: float = 50.0,
    t_min: float = 1e-2,
    jitter: float = 1e-6,
    progress: bool = False,
):
    """FlowGP gradient-free guidance driven by a Python for-loop.

    Identical update to the `lax.scan` samplers in `blackbox_conditioning.py`
    and `sitrep_fusion.py`; pulled out into a Python loop so `score_fn` may be
    a non-jittable callback (e.g. a HuggingFace LLM forward).

    `score_fn(f0)` takes a JAX array of shape (n_samples, S, M) and returns a
    JAX array of shape (n_samples, S) -- the log-score per candidate.
    Reuses the same Fisher-identity step the original FlowGP paper uses.
    """
    from flowgp import make_schedule, snr_uniform_grid

    M = int(m_pred.shape[0])
    beta, alpha, log_snr = make_schedule()
    L = jnp.linalg.cholesky(K_pred + jitter * jnp.eye(M))
    ts = snr_uniform_grid(log_snr, T, t_min=t_min)
    dts = ts[:-1] - ts[1:]
    alphas = alpha(ts[:-1])
    betas = beta(ts[:-1])

    key, k0 = jax.random.split(key)
    f_hat = jax.random.normal(k0, (n_samples, M))

    iterator = range(T)
    if progress:
        try:
            from tqdm import tqdm

            iterator = tqdm(iterator, desc="FlowGP-LLM")
        except ImportError:
            pass

    for i in iterator:
        a, b, dt = alphas[i], betas[i], dts[i]
        key, ke = jax.random.split(key)
        eps = jax.random.normal(ke, (n_samples, S, M))
        f0_hat = a * f_hat[:, None, :] + jnp.sqrt(1.0 - a**2) * eps
        f0 = f0_hat @ L.T + m_pred
        scores = score_fn(f0)  # the only non-JAX bit
        w = jax.nn.softmax(scores, axis=1)
        e_c = jnp.einsum("ns,nsm->nm", w, f0_hat)
        v = 0.5 * b * a / (1.0 - a**2 + 1e-6) * (e_c - a * f_hat)
        nrm = jnp.linalg.norm(v, axis=1, keepdims=True)
        v = v_max * jnp.tanh(nrm / v_max) / (nrm + 1e-8) * v
        f_hat = f_hat + dt * v

    return f_hat @ L.T + m_pred


__all__ = [
    "LLMVerifier",
    "MockVerifier",
    "make_score_fn",
    "render_trajectory",
    "debug_show_prompt",
    "build_yesno_prompt",
    "build_seqlp_context",
    "flowgp_pyloop",
]


