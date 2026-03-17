import asyncio
import os
import re
import tempfile
import threading
from typing import Any
from urllib.parse import urlparse
import requests
import urllib3
from flask import Flask, jsonify, render_template, request
from urllib3.exceptions import InsecureRequestWarning
try:
    import opengradient as og
except Exception:
    og = None


app = Flask(__name__)

REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "45"))
DEFAULT_PROVIDER = os.getenv("INFERENCE_PROVIDER", "auto").lower()
DEFAULT_MODEL = os.getenv("MODEL_NAME", "gpt-4o-mini")
SYSTEM_PROMPT = os.getenv(
    "SYSTEM_PROMPT",
    "You are a helpful AI assistant. Keep answers clear and concise.",
)
DEFAULT_CHAIN_ID = os.getenv("CHAIN_ID", "84532")
DEFAULT_CHAIN_NAME = os.getenv("CHAIN_NAME", "Base Sepolia")
DEFAULT_RPC_URL = os.getenv("RPC_URL", "https://sepolia.base.org")
DEFAULT_OPG_TOKEN = os.getenv("OPG_TOKEN_ADDRESS", "")
DEFAULT_OG_SPENDER = os.getenv("OG_SPENDER_ADDRESS", "")
DEFAULT_OPG_FAUCET_URL = os.getenv("OPG_FAUCET_URL", "")
DEFAULT_ETH_FAUCET_URL = os.getenv("ETH_FAUCET_URL", "")

OG_SDK_MODEL = os.getenv("OG_SDK_MODEL", "GPT_4_1_2025_04_14")
OG_SETTLEMENT_MODE = os.getenv("OG_SETTLEMENT_MODE", "PRIVATE").upper()
OG_APPROVAL_OPG_AMOUNT = float(os.getenv("OG_APPROVAL_OPG_AMOUNT", "5"))

X402_ENDPOINT = os.getenv("X402_ENDPOINT", "https://llm.opengradient.ai/v1/chat/completions")
X402_DEFAULT_MODEL = os.getenv("X402_DEFAULT_MODEL", "google/gemini-2.5-flash")
X402_DEFAULT_SETTLEMENT = os.getenv("X402_DEFAULT_SETTLEMENT", "private")
X402_FALLBACK_ENDPOINTS = os.getenv("X402_FALLBACK_ENDPOINTS", "https://13.59.207.188/v1/chat/completions")

_approval_lock = threading.Lock()
urllib3.disable_warnings(InsecureRequestWarning)

def _json_error(message: str, status: int = 500, details: Any | None = None):
    payload = {"error": message}
    if details is not None:
        payload["details"] = details
    return jsonify(payload), status


def _run_async(coro):
    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(coro)
    finally:
        loop.close()
        asyncio.set_event_loop(None)


def _extract_x402_headers(headers: requests.structures.CaseInsensitiveDict) -> dict[str, str]:
    out = {}
    for key, value in headers.items():
        lk = key.lower()
        if lk.startswith("x-") or lk.startswith("payment") or lk.startswith("www-authenticate"):
            out[key] = value
    return out




def _get_x402_candidate_endpoints() -> list[str]:
    candidates = [X402_ENDPOINT]
    for endpoint in X402_FALLBACK_ENDPOINTS.split(","):
        endpoint = endpoint.strip()
        if endpoint:
            candidates.append(endpoint)

    deduped: list[str] = []
    for endpoint in candidates:
        if endpoint not in deduped:
            deduped.append(endpoint)
    return deduped


def _post_x402_with_fallback(headers: dict[str, str], payload: dict[str, Any]) -> tuple[requests.Response, str]:
    last_exc: Exception | None = None
    for endpoint in _get_x402_candidate_endpoints():
        try:
            host = (urlparse(endpoint).hostname or "").strip()
            is_ipv4 = bool(re.match(r"^\d{1,3}(\.\d{1,3}){3}$", host))
            response = requests.post(
                endpoint,
                headers=headers,
                json=payload,
                timeout=REQUEST_TIMEOUT,
                verify=not is_ipv4,
            )
            return response, endpoint
        except requests.RequestException as exc:
            msg = str(exc).lower()
            dns_related = (
                "failed to resolve" in msg
                or "name or service not known" in msg
                or "temporary failure in name resolution" in msg
                or "no address associated with hostname" in msg
            )
            if dns_related:
                last_exc = exc
                continue
            raise

    if last_exc is not None:
        raise last_exc
    raise RuntimeError("x402 request failed: no endpoints available")

def _is_402_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "402" in msg and "payment required" in msg


def _x402_prepare_request(
    messages: list[dict[str, str]],
    model: str | None = None,
    max_tokens: int = 300,
    settlement: str | None = None,
) -> tuple[int, dict[str, str], Any, str]:
    headers = {
        "Content-Type": "application/json",
        "X-SETTLEMENT-TYPE": (settlement or X402_DEFAULT_SETTLEMENT).strip().lower(),
    }
    api_key = os.getenv("OG_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    response, endpoint_used = _post_x402_with_fallback(
        headers=headers,
        payload={
            "model": (model or X402_DEFAULT_MODEL).strip(),
            "messages": messages,
            "max_tokens": max_tokens,
        },
    )

    body: Any
    try:
        body = response.json()
    except Exception:
        body = response.text
    return response.status_code, _extract_x402_headers(response.headers), body, endpoint_used


def _resolve_og_model():
    if og is None:
        raise RuntimeError("opengradient package is not available")

    model = getattr(og.TEE_LLM, OG_SDK_MODEL, None)
    if model is None:
        raise RuntimeError(
            f"Unknown OG_SDK_MODEL '{OG_SDK_MODEL}'. Example: GPT_5, GPT_4_1_2025_04_14, GEMINI_2_5_FLASH"
        )
    return model


def _resolve_settlement_mode():
    if og is None:
        raise RuntimeError("opengradient package is not available")

    mode = getattr(og.x402SettlementMode, OG_SETTLEMENT_MODE, None)
    if mode is None:
        raise RuntimeError(
            f"Unknown OG_SETTLEMENT_MODE '{OG_SETTLEMENT_MODE}'. Use PRIVATE, BATCH_HASHED, or INDIVIDUAL_FULL"
        )
    return mode


def _ensure_approval_once(llm):
    with _approval_lock:
        llm.ensure_opg_approval(opg_amount=OG_APPROVAL_OPG_AMOUNT)


def _get_hub():
    if og is None:
        raise RuntimeError("opengradient package is not installed")

    email = os.getenv("OG_HUB_EMAIL")
    password = os.getenv("OG_HUB_PASSWORD")
    if not email or not password:
        raise RuntimeError("Set OG_HUB_EMAIL and OG_HUB_PASSWORD in Railway variables")

    return og.ModelHub(email=email, password=password)


def _get_alpha():
    if og is None:
        raise RuntimeError("opengradient package is not installed")

    private_key = os.getenv("OG_ALPHA_PRIVATE_KEY") or os.getenv("OG_PRIVATE_KEY")
    if not private_key:
        raise RuntimeError("Set OG_ALPHA_PRIVATE_KEY (or OG_PRIVATE_KEY) in Railway variables")

    return og.Alpha(private_key=private_key)


async def call_opengradient_sdk_async(prompt: str) -> str:
    if og is None:
        raise RuntimeError("opengradient package is not installed")

    private_key = os.getenv("OG_PRIVATE_KEY")
    if not private_key:
        raise RuntimeError("OG_PRIVATE_KEY is not set")

    llm = og.LLM(private_key=private_key)
    _ensure_approval_once(llm)

    result = await llm.chat(
        model=_resolve_og_model(),
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        max_tokens=300,
        temperature=0.7,
        x402_settlement_mode=_resolve_settlement_mode(),
    )

    content = ""
    if isinstance(result.chat_output, dict):
        content = (result.chat_output.get("content") or "").strip()

    if not content:
        raise RuntimeError("OpenGradient SDK returned empty content")

    return content


def call_opengradient_sdk(prompt: str) -> str:
    return _run_async(call_opengradient_sdk_async(prompt))



def call_opengradient_sdk_with_x402_fallback(prompt: str) -> tuple[str, str]:
    try:
        return call_opengradient_sdk(prompt), "opengradient_sdk"
    except Exception as exc:
        if not _is_402_error(exc):
            raise

        status_code, headers, body, endpoint_used = _x402_prepare_request(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            model=X402_DEFAULT_MODEL,
            max_tokens=300,
            settlement=X402_DEFAULT_SETTLEMENT,
        )

        if status_code == 200:
            try:
                content = body["choices"][0]["message"]["content"].strip()
                if content:
                    return content, "x402_gateway"
            except Exception:
                pass

        if status_code == 402:
            requirement_preview = str(headers)[:400]
            message = (
                "SDK returned 402. x402 payment is required. "
                "Use the Raw x402 Gateway block: click Prepare, sign payment payload, "
                "paste X-PAYMENT, then click Submit. "
                f"Payment headers: {requirement_preview}. Endpoint used: {endpoint_used}"
            )
            return message, "x402_prepare_required"

        raise RuntimeError(f"x402 fallback failed with status {status_code}: {str(body)[:400]}")


def call_openai(prompt: str) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")

    response = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": os.getenv("OPENAI_MODEL", DEFAULT_MODEL),
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.7,
            "max_tokens": 300,
        },
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"].strip()


def call_gemini(prompt: str) -> str:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set")

    model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
    response = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}",
        headers={"Content-Type": "application/json"},
        json={
            "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.7, "maxOutputTokens": 300},
        },
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    data = response.json()
    candidates = data.get("candidates") or []
    if not candidates:
        raise RuntimeError("Gemini returned no candidates")
    parts = candidates[0].get("content", {}).get("parts", [])
    text = "".join(part.get("text", "") for part in parts).strip()
    if not text:
        raise RuntimeError("Gemini returned an empty response")
    return text


def call_opengradient_http(prompt: str) -> str:
    endpoint = os.getenv(
        "OG_ENDPOINT",
        "https://llm.opengradient.ai/v1/chat/completions",
    )
    api_key = os.getenv("OG_API_KEY")
    if not api_key:
        raise RuntimeError("OG_API_KEY is not set")

    response = requests.post(
        endpoint,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": os.getenv("OG_MODEL", "google/gemini-2.5-flash"),
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": 300,
        },
        timeout=REQUEST_TIMEOUT,
    )

    if response.status_code == 402:
        raise RuntimeError(
            "OpenGradient returned 402 Payment Required. Use raw x402 panel or SDK/fallback mode."
        )

    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"].strip()


def generate_reply(prompt: str) -> tuple[str, str]:
    provider = DEFAULT_PROVIDER
    errors: list[str] = []

    if provider == "openai":
        return call_openai(prompt), "openai"
    if provider == "gemini":
        return call_gemini(prompt), "gemini"
    if provider == "opengradient":
        return call_opengradient_http(prompt), "opengradient"
    if provider == "opengradient_sdk":
        return call_opengradient_sdk_with_x402_fallback(prompt)
    for name, fn in (
        ("opengradient_sdk", lambda p: call_opengradient_sdk_with_x402_fallback(p)[0]),
        ("openai", call_openai),
        ("gemini", call_gemini),
        ("opengradient", call_opengradient_http),
    ):
        try:
            return fn(prompt), name
        except Exception as exc:
            errors.append(f"{name}: {exc}")

    raise RuntimeError("All providers failed: " + " | ".join(errors))


@app.get("/health")
def health():
    return jsonify(
        {
            "ok": True,
            "provider": DEFAULT_PROVIDER,
            "has_openai": bool(os.getenv("OPENAI_API_KEY")),
            "has_gemini": bool(os.getenv("GEMINI_API_KEY")),
            "has_og": bool(os.getenv("OG_API_KEY")),
            "has_og_private_key": bool(os.getenv("OG_PRIVATE_KEY")),
            "has_alpha_private_key": bool(os.getenv("OG_ALPHA_PRIVATE_KEY") or os.getenv("OG_PRIVATE_KEY")),
            "og_sdk_available": og is not None,
            "og_sdk_model": OG_SDK_MODEL,
            "og_settlement_mode": OG_SETTLEMENT_MODE,
            "model_hub_configured": bool(os.getenv("OG_HUB_EMAIL") and os.getenv("OG_HUB_PASSWORD")),
            "x402_endpoint": X402_ENDPOINT,
            "x402_fallback_endpoints": X402_FALLBACK_ENDPOINTS,
        }
    )


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/config")
def config():
    return jsonify(
        {
            "chainId": DEFAULT_CHAIN_ID,
            "chainName": DEFAULT_CHAIN_NAME,
            "rpcUrl": DEFAULT_RPC_URL,
            "opgTokenAddress": DEFAULT_OPG_TOKEN,
            "ogSpenderAddress": DEFAULT_OG_SPENDER,
            "opgFaucetUrl": DEFAULT_OPG_FAUCET_URL,
            "ethFaucetUrl": DEFAULT_ETH_FAUCET_URL,
            "modelHubConfigured": bool(os.getenv("OG_HUB_EMAIL") and os.getenv("OG_HUB_PASSWORD")),
            "alphaConfigured": bool(os.getenv("OG_ALPHA_PRIVATE_KEY") or os.getenv("OG_PRIVATE_KEY")),
            "x402Endpoint": X402_ENDPOINT,
            "x402FallbackEndpoints": X402_FALLBACK_ENDPOINTS,
            "x402DefaultModel": X402_DEFAULT_MODEL,
            "x402DefaultSettlement": X402_DEFAULT_SETTLEMENT,
        }
    )


@app.post("/chat")
def chat():
    data = request.get_json(silent=True) or {}
    prompt = (data.get("prompt") or "").strip()

    if not prompt:
        return _json_error("Prompt is required", 400)

    try:
        reply, provider = generate_reply(prompt)
        return jsonify({"response": reply, "provider": provider})
    except requests.HTTPError as exc:
        details = exc.response.text[:500] if exc.response is not None else str(exc)
        return _json_error("Upstream API error", 502, details)
    except requests.RequestException as exc:
        return _json_error("Network error while calling provider", 502, str(exc))
    except Exception as exc:
        return _json_error(str(exc), 500)


@app.post("/api/x402/prepare")
def x402_prepare():
    data = request.get_json(silent=True) or {}
    model = (data.get("model") or X402_DEFAULT_MODEL).strip()
    settlement = (data.get("settlement") or X402_DEFAULT_SETTLEMENT).strip().lower()
    max_tokens = int(data.get("max_tokens") or 256)
    messages = data.get("messages")

    if not isinstance(messages, list) or not messages:
        return _json_error("messages must be a non-empty array", 400)

    headers = {
        "Content-Type": "application/json",
        "X-SETTLEMENT-TYPE": settlement,
    }

    api_key = os.getenv("OG_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        response, endpoint_used = _post_x402_with_fallback(
            headers=headers,
            payload={
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
            },
        )

        body = None
        try:
            body = response.json()
        except Exception:
            body = response.text

        return jsonify(
            {
                "ok": response.status_code in (200, 402),
                "status_code": response.status_code,
                "endpoint_used": endpoint_used,
                "headers": _extract_x402_headers(response.headers),
                "body": body,
                "hint": "If status_code is 402, sign payment payload client-side and call /api/x402/submit with x_payment",
            }
        ), (200 if response.status_code in (200, 402) else 502)
    except requests.RequestException as exc:
        return _json_error("x402 prepare request failed", 502, str(exc))


@app.post("/api/x402/submit")
def x402_submit():
    data = request.get_json(silent=True) or {}
    model = (data.get("model") or X402_DEFAULT_MODEL).strip()
    settlement = (data.get("settlement") or X402_DEFAULT_SETTLEMENT).strip().lower()
    max_tokens = int(data.get("max_tokens") or 256)
    x_payment = (data.get("x_payment") or "").strip()
    messages = data.get("messages")

    if not x_payment:
        return _json_error("x_payment is required", 400)
    if not isinstance(messages, list) or not messages:
        return _json_error("messages must be a non-empty array", 400)

    headers = {
        "Content-Type": "application/json",
        "X-SETTLEMENT-TYPE": settlement,
        "X-PAYMENT": x_payment,
    }

    api_key = os.getenv("OG_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        response, endpoint_used = _post_x402_with_fallback(
            headers=headers,
            payload={
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
            },
        )

        body = None
        try:
            body = response.json()
        except Exception:
            body = response.text

        return jsonify(
            {
                "ok": response.status_code == 200,
                "status_code": response.status_code,
                "endpoint_used": endpoint_used,
                "headers": _extract_x402_headers(response.headers),
                "body": body,
            }
        ), (200 if response.status_code == 200 else 502)
    except requests.RequestException as exc:
        return _json_error("x402 submit request failed", 502, str(exc))


@app.post("/api/modelhub/create-model")
def modelhub_create_model():
    data = request.get_json(silent=True) or {}
    model_name = (data.get("model_name") or "").strip()
    model_desc = (data.get("model_desc") or "Created from OpenGradient Neon AI Terminal").strip()

    if not model_name:
        return _json_error("model_name is required", 400)

    try:
        hub = _get_hub()
        result = hub.create_model(model_name=model_name, model_desc=model_desc)
        return jsonify({"ok": True, "model_name": model_name, "result": str(result)})
    except Exception as exc:
        return _json_error("Model create failed", 500, str(exc))


@app.post("/api/modelhub/create-version")
def modelhub_create_version():
    data = request.get_json(silent=True) or {}
    model_name = (data.get("model_name") or "").strip()
    notes = (data.get("notes") or "New version from web terminal").strip()

    if not model_name:
        return _json_error("model_name is required", 400)

    try:
        hub = _get_hub()
        version = hub.create_version(model_name=model_name, notes=notes)
        return jsonify({"ok": True, "model_name": model_name, "version": str(version)})
    except Exception as exc:
        return _json_error("Version create failed", 500, str(exc))


@app.get("/api/modelhub/list-files")
def modelhub_list_files():
    model_name = (request.args.get("model_name") or "").strip()
    version = (request.args.get("version") or "").strip()

    if not model_name or not version:
        return _json_error("model_name and version are required", 400)

    try:
        hub = _get_hub()
        files = hub.list_files(model_name=model_name, version=version)
        return jsonify({"ok": True, "model_name": model_name, "version": version, "files": files})
    except Exception as exc:
        return _json_error("List files failed", 500, str(exc))


@app.post("/api/modelhub/upload")
def modelhub_upload_file():
    model_name = (request.form.get("model_name") or "").strip()
    version = (request.form.get("version") or "").strip()
    file_obj = request.files.get("file")

    if not model_name or not version:
        return _json_error("model_name and version are required", 400)
    if file_obj is None or not file_obj.filename:
        return _json_error("file is required", 400)

    temp_path = None
    try:
        suffix = os.path.splitext(file_obj.filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            file_obj.save(tmp.name)
            temp_path = tmp.name

        hub = _get_hub()
        result = hub.upload(model_path=temp_path, model_name=model_name, version=version)
        return jsonify(
            {
                "ok": True,
                "model_name": model_name,
                "version": version,
                "file": file_obj.filename,
                "result": str(result),
            }
        )
    except Exception as exc:
        return _json_error("Upload failed", 500, str(exc))
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


@app.post("/api/alpha/infer")
def alpha_infer():
    data = request.get_json(silent=True) or {}
    model_cid = (data.get("model_cid") or "").strip()
    mode = (data.get("mode") or "VANILLA").upper().strip()
    model_input = data.get("model_input")
    max_retries = data.get("max_retries")

    if not model_cid:
        return _json_error("model_cid is required", 400)
    if not isinstance(model_input, dict):
        return _json_error("model_input must be a JSON object", 400)

    try:
        alpha = _get_alpha()
        inference_mode = getattr(og.InferenceMode, mode, None)
        if inference_mode is None:
            return _json_error("mode must be one of: VANILLA, TEE, ZKML", 400)

        kwargs = {
            "model_cid": model_cid,
            "inference_mode": inference_mode,
            "model_input": model_input,
        }
        if max_retries is not None:
            kwargs["max_retries"] = int(max_retries)

        result = alpha.infer(**kwargs)
        output = getattr(result, "model_output", None)
        tx_hash = getattr(result, "tx_hash", None)
        return jsonify(
            {
                "ok": True,
                "model_cid": model_cid,
                "mode": mode,
                "model_output": output,
                "tx_hash": tx_hash,
                "raw": str(result),
            }
        )
    except Exception as exc:
        return _json_error("Alpha inference failed", 500, str(exc))


@app.post("/api/alpha/new-workflow")
def alpha_new_workflow():
    data = request.get_json(silent=True) or {}

    model_cid = (data.get("model_cid") or "").strip()
    input_tensor_name = (data.get("input_tensor_name") or "open_high_low_close").strip()

    query = data.get("input_query") or {}
    scheduler = data.get("scheduler") or {}

    if not model_cid:
        return _json_error("model_cid is required", 400)

    try:
        if og is None:
            raise RuntimeError("opengradient package is not installed")

        from opengradient.types import CandleOrder, CandleType, HistoricalInputQuery, SchedulerParams

        candle_types_input = query.get("candle_types") or ["OPEN", "HIGH", "LOW", "CLOSE"]
        candle_types = []
        for c in candle_types_input:
            ct = getattr(CandleType, str(c).upper(), None)
            if ct is None:
                raise RuntimeError(f"Unknown candle type: {c}")
            candle_types.append(ct)

        order = getattr(CandleOrder, str(query.get("order", "ASCENDING")).upper(), None)
        if order is None:
            raise RuntimeError("order must be ASCENDING or DESCENDING")

        input_query = HistoricalInputQuery(
            base=str(query.get("base", "ETH")),
            quote=str(query.get("quote", "USD")),
            total_candles=int(query.get("total_candles", 10)),
            candle_duration_in_mins=int(query.get("candle_duration_in_mins", 30)),
            order=order,
            candle_types=candle_types,
        )

        scheduler_params = SchedulerParams(
            frequency=int(scheduler.get("frequency", 3600)),
            duration_hours=int(scheduler.get("duration_hours", 24)),
        )

        alpha = _get_alpha()
        contract_address = alpha.new_workflow(
            model_cid=model_cid,
            input_query=input_query,
            input_tensor_name=input_tensor_name,
            scheduler_params=scheduler_params,
        )
        return jsonify({"ok": True, "contract_address": str(contract_address)})
    except Exception as exc:
        return _json_error("New workflow deployment failed", 500, str(exc))


@app.post("/api/alpha/run-workflow")
def alpha_run_workflow():
    data = request.get_json(silent=True) or {}
    contract_address = (data.get("contract_address") or "").strip()
    if not contract_address:
        return _json_error("contract_address is required", 400)

    try:
        alpha = _get_alpha()
        result = alpha.run_workflow(contract_address)
        return jsonify({"ok": True, "contract_address": contract_address, "result": str(result)})
    except Exception as exc:
        return _json_error("Run workflow failed", 500, str(exc))


@app.post("/api/alpha/read-workflow-result")
def alpha_read_workflow_result():
    data = request.get_json(silent=True) or {}
    contract_address = (data.get("contract_address") or "").strip()
    if not contract_address:
        return _json_error("contract_address is required", 400)

    try:
        alpha = _get_alpha()
        result = alpha.read_workflow_result(contract_address)
        return jsonify({"ok": True, "contract_address": contract_address, "result": str(result)})
    except Exception as exc:
        return _json_error("Read workflow result failed", 500, str(exc))


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8080"))
    app.run(host="0.0.0.0", port=port)









