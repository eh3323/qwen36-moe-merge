"""Put on PYTHONPATH for `asibench score` only (scripts/eval_asi_score.sh).

The ASI judges request temperature=0, which the official judge model (gpt-5.5) rejects at the API: "Unsupported value:
'temperature' does not support 0.0 with this model. Only the default (1) value is supported." (2026-10-15: 8/60 Base
instances evaluation_invalid.) LiteLLM's drop_params does not drop it for this model, so the temperature is removed
here from every gpt-5* LiteLLM call (plain kwarg or extra_body) and the judge runs at the API default, as it must
behind any gateway that accepts the request. Nothing else in the request changes. Python processes without LiteLLM
(task scorer environments) are unaffected.
"""
try:
    import functools

    import litellm

    def _strip(kwargs, args):
        model = str(kwargs.get("model") or (args[0] if args else ""))
        if "gpt-5" in model:
            kwargs.pop("temperature", None)
            extra = kwargs.get("extra_body")
            if isinstance(extra, dict):
                extra.pop("temperature", None)
        return kwargs

    def _wrap(fn):
        @functools.wraps(fn)
        def call(*args, **kwargs):
            return fn(*args, **_strip(kwargs, args))
        return call

    def _awrap(fn):
        @functools.wraps(fn)
        async def call(*args, **kwargs):
            return await fn(*args, **_strip(kwargs, args))
        return call

    litellm.completion = _wrap(litellm.completion)
    litellm.acompletion = _awrap(litellm.acompletion)
except Exception:
    pass
