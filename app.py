import asyncio
import base64
import json
import os
import re
import tempfile
import threading
import time
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlparse
import requests
import urllib3
from flask import Flask, jsonify, render_template, request
from urllib3.exceptions import InsecureRequestWarning

X402_IMPORT_ERRORS: list[str] = []
try:
    from eth_account import Account as EthAccount
except Exception:
    EthAccount = None

try:
    from x402.clients.base import x402Client as X402Client
except Exception:
    try:
        from x402.clients.requests import x402Client as X402Client
    except Exception:
        X402Client = None

try:
    from x402 import x402ClientSync as X402ClientSync
except Exception as exc:
    X402ClientSync = None
    X402_IMPORT_ERRORS.append(f"x402ClientSync: {exc}")

try:
    # Import directly from client submodule to avoid optional facilitator/web3 deps.
    from x402.mechanisms.evm.exact.client import ExactEvmScheme as X402ExactEvmClientScheme
except Exception as exc:
    X402ExactEvmClientScheme = None
    X402_IMPORT_ERRORS.append(f"ExactEvmScheme(client): {exc}")

try:
    from x402.http import (
        encode_payment_signature_header,
        PAYMENT_REQUIRED_HEADER,
        PAYMENT_SIGNATURE_HEADER,
        X_PAYMENT_HEADER,
        decode_payment_required_header,
    )
except Exception as exc:
    encode_payment_signature_header = None
    PAYMENT_REQUIRED_HEADER = "Payment-Required"
    PAYMENT_SIGNATURE_HEADER = "PAYMENT-SIGNATURE"
    X_PAYMENT_HEADER = "X-PAYMENT"
    decode_payment_required_header = None
    X402_IMPORT_ERRORS.append(f"x402.http: {exc}")
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
DEFAULT_OPG_FAUCET_URL = os.getenv("OPG_FAUCET_URL", "https://faucet.opengradient.ai")
DEFAULT_ETH_FAUCET_URL = os.getenv("ETH_FAUCET_URL", "https://www.alchemy.com/faucets/base-sepolia")
GITHUB_API_BASE = os.getenv("GITHUB_API_BASE", "https://api.github.com")
REPO_CHECK_README_MAX_CHARS = int(os.getenv("REPO_CHECK_README_MAX_CHARS", "5000"))

OG_SDK_MODEL = os.getenv("OG_SDK_MODEL", "GPT_4_1_2025_04_14")
OG_SETTLEMENT_MODE = os.getenv("OG_SETTLEMENT_MODE", "PRIVATE").upper()
OG_APPROVAL_OPG_AMOUNT = float(os.getenv("OG_APPROVAL_OPG_AMOUNT", "5"))

X402_ENDPOINT = os.getenv("X402_ENDPOINT", "https://llm.opengradient.ai/v1/chat/completions")
X402_DEFAULT_MODEL = os.getenv("X402_DEFAULT_MODEL", "google/gemini-2.5-flash")
X402_DEFAULT_SETTLEMENT = os.getenv("X402_DEFAULT_SETTLEMENT", "private")
X402_FALLBACK_ENDPOINTS = os.getenv("X402_FALLBACK_ENDPOINTS", "https://13.59.207.188/v1/chat/completions")
ENABLE_WIKI_FALLBACK = os.getenv("ENABLE_WIKI_FALLBACK", "true").strip().lower() in ("1", "true", "yes", "on")
BUILD_MARKER = "no_manual_x402_v2"

_approval_lock = threading.Lock()
_x402_backend_approval_ready = False
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


def _get_header_case_insensitive(headers: dict[str, str], target_name: str) -> str | None:
    target = target_name.strip().lower()
    for key, value in headers.items():
        normalized_key = key.strip().lower().replace("_", "-")
        if normalized_key == target:
            return value
        if target == "payment-required" and "payment-required" in normalized_key:
            return value
    return None


def _get_payment_required_from_body(body: Any) -> str | None:
    if not isinstance(body, dict):
        return None

    # Some gateways/proxies can forward payment requirements inside JSON body.
    for key, value in body.items():
        k = str(key).strip().lower().replace("_", "-")
        if "payment-required" in k and isinstance(value, str) and value.strip():
            return value.strip()
    return None




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


def _pick_payment_requirement(payment_required):
    accepts = payment_required.accepts or []
    if not accepts:
        raise RuntimeError("No payment requirements returned by x402 gateway")

    preferred_network = os.getenv("X402_PREFERRED_NETWORK", "eip155:84532").strip()
    preferred_asset = (DEFAULT_OPG_TOKEN or "").strip().lower()

    for req in accepts:
        if req.network == preferred_network and (not preferred_asset or req.asset.lower() == preferred_asset):
            return req
    for req in accepts:
        if req.network == preferred_network:
            return req
    for req in accepts:
        if preferred_asset and req.asset.lower() == preferred_asset:
            return req
    return accepts[0]


def _parse_chain_id(network: str) -> int:
    value = str(network or "").strip()
    if ":" in value:
        value = value.split(":")[-1]
    return int(value, 10)


class _LocalEthAccountSigner:
    def __init__(self, account):
        self._account = account

    @property
    def address(self) -> str:
        return self._account.address

    def sign_typed_data(
        self,
        domain,
        types,
        primary_type,
        message,
    ) -> bytes:
        types_dict: dict[str, list[dict[str, str]]] = {}
        for type_name, fields in types.items():
            parsed_fields = []
            for field in fields:
                if isinstance(field, dict):
                    parsed_fields.append(
                        {
                            "name": str(field.get("name", "")),
                            "type": str(field.get("type", "")),
                        }
                    )
                else:
                    parsed_fields.append(
                        {
                            "name": str(getattr(field, "name", "")),
                            "type": str(getattr(field, "type", "")),
                        }
                    )
            types_dict[type_name] = parsed_fields

        domain_dict = domain
        if not isinstance(domain, dict):
            domain_dict = {
                "name": getattr(domain, "name", None),
                "version": getattr(domain, "version", None),
                "chainId": getattr(domain, "chain_id", None),
                "verifyingContract": getattr(domain, "verifying_contract", None),
            }

        signed = self._account.sign_typed_data(
            domain_data=domain_dict,
            message_types=types_dict,
            message_data=message,
        )
        return bytes(signed.signature)


def _build_legacy_xpayment_header(payment_required_header: str) -> str:
    if EthAccount is None or decode_payment_required_header is None:
        raise RuntimeError("legacy x402 signer dependencies are unavailable")

    private_key = os.getenv("OG_PRIVATE_KEY")
    if not private_key:
        raise RuntimeError("OG_PRIVATE_KEY is required for x402 payment signing")

    payment_required = decode_payment_required_header(payment_required_header)
    selected = _pick_payment_requirement(payment_required)
    account = EthAccount.from_key(private_key)

    network = str(getattr(selected, "network", "") or "")
    pay_to = str(getattr(selected, "pay_to", None) or getattr(selected, "payTo", ""))
    amount = str(
        getattr(selected, "max_amount_required", None)
        or getattr(selected, "amount", None)
        or "0"
    )
    max_timeout = int(getattr(selected, "max_timeout_seconds", 600) or 600)
    extra = getattr(selected, "extra", {}) or {}

    if not pay_to or amount == "0":
        raise RuntimeError("legacy x402 signer could not resolve pay_to/amount")

    valid_after = int(time.time()) - 60
    valid_before = int(time.time()) + max_timeout
    nonce_hex = os.urandom(32).hex()

    signed = account.sign_typed_data(
        domain_data={
            "name": str(extra.get("name") or "OPG"),
            "version": str(extra.get("version") or "1"),
            "chainId": _parse_chain_id(network),
            "verifyingContract": str(getattr(selected, "asset", "")),
        },
        message_types={
            "TransferWithAuthorization": [
                {"name": "from", "type": "address"},
                {"name": "to", "type": "address"},
                {"name": "value", "type": "uint256"},
                {"name": "validAfter", "type": "uint256"},
                {"name": "validBefore", "type": "uint256"},
                {"name": "nonce", "type": "bytes32"},
            ]
        },
        message_data={
            "from": account.address,
            "to": pay_to,
            "value": int(amount),
            "validAfter": valid_after,
            "validBefore": valid_before,
            "nonce": bytes.fromhex(nonce_hex),
        },
    )

    signature = signed.signature.hex()
    if not signature.startswith("0x"):
        signature = f"0x{signature}"

    payload = {
        "x402Version": int(getattr(payment_required, "x402_version", 2) or 2),
        "scheme": str(getattr(selected, "scheme", "upto") or "upto"),
        "network": network,
        "payload": {
            "signature": signature,
            "authorization": {
                "from": account.address,
                "to": pay_to,
                "value": amount,
                "validAfter": str(valid_after),
                "validBefore": str(valid_before),
                "nonce": f"0x{nonce_hex}",
            },
        },
    }

    return base64.b64encode(
        json.dumps(payload, separators=(",", ":")).encode("utf-8")
    ).decode("utf-8")


def _sign_payment_required_header(payment_required_header: str) -> str:
    if EthAccount is None or decode_payment_required_header is None:
        raise RuntimeError(
            "x402 signer dependencies are unavailable in this runtime "
            f"(EthAccount={EthAccount is not None}, decode_payment_required_header={decode_payment_required_header is not None})"
        )

    private_key = os.getenv("OG_PRIVATE_KEY")
    if not private_key:
        raise RuntimeError("OG_PRIVATE_KEY is required for x402 payment signing")

    payment_required = decode_payment_required_header(payment_required_header)
    account = EthAccount.from_key(private_key)

    # Compatibility path for older x402 APIs that expose x402.clients.*
    if X402Client is not None:
        try:
            selected = _pick_payment_requirement(payment_required)
            x402_client = X402Client(account=account)
            return x402_client.create_payment_header(
                selected,
                x402_version=payment_required.x402_version,
            )
        except TypeError:
            pass
        except Exception:
            pass

    # x402>=2.0 path
    if (
        X402ClientSync is not None
        and X402ExactEvmClientScheme is not None
        and encode_payment_signature_header is not None
    ):
        selected = _pick_payment_requirement(payment_required)

        def _selector(_version, requirements):
            for req in requirements:
                if (
                    getattr(req, "network", None) == getattr(selected, "network", None)
                    and getattr(req, "asset", None) == getattr(selected, "asset", None)
                    and getattr(req, "scheme", None) == getattr(selected, "scheme", None)
                ):
                    return req
            return requirements[0]

        client = X402ClientSync(payment_requirements_selector=_selector)
        signer = _LocalEthAccountSigner(account)
        scheme = X402ExactEvmClientScheme(signer)
        if getattr(selected, "scheme", None):
            try:
                scheme.scheme = str(getattr(selected, "scheme"))
            except Exception:
                pass
        client.register(str(getattr(selected, "network", "eip155:*")), scheme)
        payload = client.create_payment_payload(payment_required)
        return encode_payment_signature_header(payload)

    # Last-resort fallback: generate legacy X-PAYMENT format.
    return _build_legacy_xpayment_header(payment_required_header)


def _needs_legacy_retry(status_code: int, body: Any) -> bool:
    if status_code < 500:
        return False
    text = str(body or "").lower()
    return (
        "facilitator verify failed" in text
        or "reading 'from'" in text
        or 'reading "from"' in text
    )


def _x402_auto_pay_request(
    messages: list[dict[str, str]],
    model: str | None = None,
    max_tokens: int = 300,
    settlement: str | None = None,
) -> tuple[int, dict[str, str], Any, str]:
    status_code, headers, body, endpoint_used = _x402_prepare_request(
        messages=messages,
        model=model,
        max_tokens=max_tokens,
        settlement=settlement,
    )
    if status_code != 402:
        return status_code, headers, body, endpoint_used

    payment_required_header = _get_header_case_insensitive(headers, PAYMENT_REQUIRED_HEADER)
    if not payment_required_header:
        for alt_name in ("Payment-Required", "PAYMENT-REQUIRED", "payment-required"):
            payment_required_header = _get_header_case_insensitive(headers, alt_name)
            if payment_required_header:
                break
    if not payment_required_header:
        payment_required_header = _get_payment_required_from_body(body)
    if not payment_required_header:
        available_keys = ",".join(list(headers.keys())[:20])
        raise RuntimeError(f"x402 returned 402 but missing PAYMENT-REQUIRED header; available headers: {available_keys}")

    payment_required = decode_payment_required_header(payment_required_header)
    signed = _sign_payment_required_header(payment_required_header)

    headers_retry = {
        "Content-Type": "application/json",
        "X-SETTLEMENT-TYPE": (settlement or X402_DEFAULT_SETTLEMENT).strip().lower(),
    }
    if payment_required.x402_version == 2:
        headers_retry[PAYMENT_SIGNATURE_HEADER] = signed
        headers_retry[X_PAYMENT_HEADER] = signed
    else:
        headers_retry[X_PAYMENT_HEADER] = signed
        headers_retry[PAYMENT_SIGNATURE_HEADER] = signed

    api_key = os.getenv("OG_API_KEY")
    if api_key:
        headers_retry["Authorization"] = f"Bearer {api_key}"

    response, endpoint_retry = _post_x402_with_fallback(
        headers=headers_retry,
        payload={
            "model": (model or X402_DEFAULT_MODEL).strip(),
            "messages": messages,
            "max_tokens": max_tokens,
        },
    )
    try:
        body_retry: Any = response.json()
    except Exception:
        body_retry = response.text

    if _needs_legacy_retry(response.status_code, body_retry):
        legacy_signed = _build_legacy_xpayment_header(payment_required_header)
        headers_retry_legacy = {
            "Content-Type": "application/json",
            "X-SETTLEMENT-TYPE": (settlement or X402_DEFAULT_SETTLEMENT).strip().lower(),
            X_PAYMENT_HEADER: legacy_signed,
            PAYMENT_SIGNATURE_HEADER: legacy_signed,
        }
        if api_key:
            headers_retry_legacy["Authorization"] = f"Bearer {api_key}"

        legacy_response, legacy_endpoint = _post_x402_with_fallback(
            headers=headers_retry_legacy,
            payload={
                "model": (model or X402_DEFAULT_MODEL).strip(),
                "messages": messages,
                "max_tokens": max_tokens,
            },
        )
        try:
            legacy_body: Any = legacy_response.json()
        except Exception:
            legacy_body = legacy_response.text
        return (
            legacy_response.status_code,
            _extract_x402_headers(legacy_response.headers),
            legacy_body,
            legacy_endpoint,
        )

    return (
        response.status_code,
        _extract_x402_headers(response.headers),
        body_retry,
        endpoint_retry,
    )


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


def _ensure_x402_backend_approval_once():
    global _x402_backend_approval_ready
    if _x402_backend_approval_ready:
        return

    if og is None:
        return

    private_key = os.getenv("OG_PRIVATE_KEY")
    if not private_key:
        return

    with _approval_lock:
        if _x402_backend_approval_ready:
            return
        llm = og.LLM(private_key=private_key)
        llm.ensure_opg_approval(opg_amount=OG_APPROVAL_OPG_AMOUNT)
        _x402_backend_approval_ready = True


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


def _build_alpha_error_details(exc: Exception, mode: str, model_cid: str) -> Any:
    message = str(exc)
    details: dict[str, Any] = {
        "message": message,
        "mode": mode,
        "model_cid": model_cid,
    }
    if "InferenceResult event not found" in message:
        details["hint"] = (
            "Alpha transaction finished but the expected InferenceResult event was not emitted. "
            "This is usually alpha-network/account/mode mismatch rather than a frontend bug."
        )
        details["checklist"] = [
            "Set OG_ALPHA_PRIVATE_KEY in Railway (preferred) instead of relying only on OG_PRIVATE_KEY.",
            "Use an account that was initialized/funded on OpenGradient alpha testnet.",
            "Try mode TEE if VANILLA keeps failing for this model CID.",
            "Set max_retries to 3 and run again.",
        ]
    return details


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
    # Prefer direct x402 flow first because SDK runtime can be unstable in some deployments.
    manual_x402_message = None
    try:
        _ensure_x402_backend_approval_once()
        status_code, headers, body, endpoint_used = _x402_auto_pay_request(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            model=X402_DEFAULT_MODEL,
            max_tokens=300,
            settlement=X402_DEFAULT_SETTLEMENT,
        )
        if status_code == 200:
            content = ""
            if isinstance(body, dict):
                try:
                    content = ((body.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
                except Exception:
                    content = ""
            if content and str(content).strip():
                return str(content).strip(), "x402_gateway_auto_paid"
            if isinstance(body, str) and body.strip():
                return body.strip(), "x402_gateway_auto_paid"
            return str(body)[:1000], "x402_gateway_auto_paid"

        if status_code == 402:
            requirement_preview = str(headers)[:400]
            manual_x402_message = (
                "Payment is required and manual x402 flow is ready. "
                "Use the Raw x402 Gateway block: click Prepare, sign payload, paste X-PAYMENT (or PAYMENT-SIGNATURE), then Submit. "
                f"Payment headers: {requirement_preview}. Endpoint used: {endpoint_used}"
            )
            raise RuntimeError("x402 returned 402")

        raise RuntimeError(f"x402 auto-pay failed with status {status_code}: {str(body)[:400]}")
    except Exception as x402_exc:
        # Fallback to SDK only if direct x402 failed.
        try:
            return call_opengradient_sdk(prompt), "opengradient_sdk"
        except Exception as sdk_exc:
            debug_details = f"x402_error={x402_exc}; sdk_error={sdk_exc}"
            if manual_x402_message and ENABLE_WIKI_FALLBACK:
                try:
                    return call_wikipedia_fallback(prompt), "wikipedia_fallback"
                except Exception:
                    return call_offline_fallback(prompt), "offline_fallback"

            if manual_x402_message:
                if ENABLE_WIKI_FALLBACK:
                    return call_offline_fallback(prompt), "offline_fallback"
                raise RuntimeError(f"{manual_x402_message}. Details: {debug_details}")

            if _is_402_error(sdk_exc):
                try:
                    status_code, headers, _body, endpoint_used = _x402_prepare_request(
                        messages=[
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": prompt},
                        ],
                        model=X402_DEFAULT_MODEL,
                        max_tokens=300,
                        settlement=X402_DEFAULT_SETTLEMENT,
                    )
                    if status_code == 402:
                        if ENABLE_WIKI_FALLBACK:
                            return call_offline_fallback(prompt), "offline_fallback"
                        requirement_preview = str(headers)[:400]
                        raise RuntimeError(
                            "SDK returned 402 and x402 prepare also returned 402. "
                            f"Payment headers: {requirement_preview}. Endpoint used: {endpoint_used}. "
                            f"Details: {debug_details}"
                        )
                except Exception:
                    pass

            if ENABLE_WIKI_FALLBACK:
                return call_offline_fallback(prompt), "offline_fallback"
            raise RuntimeError(f"OpenGradient inference failed without fallback. Details: {debug_details}")


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


def call_wikipedia_fallback(prompt: str) -> str:
    headers = {
        "User-Agent": "OpenGradientTerminal/1.0 (+https://opengradient-playground-production.up.railway.app)",
    }
    search_resp = requests.get(
        "https://en.wikipedia.org/w/api.php",
        params={
            "action": "opensearch",
            "search": prompt,
            "limit": 1,
            "namespace": 0,
            "format": "json",
        },
        headers=headers,
        timeout=REQUEST_TIMEOUT,
    )
    search_resp.raise_for_status()
    search_data = search_resp.json()
    titles = search_data[1] if isinstance(search_data, list) and len(search_data) > 1 else []
    if not titles:
        raise RuntimeError("No Wikipedia match found for the prompt")

    title = str(titles[0]).strip()
    summary_resp = requests.get(
        f"https://en.wikipedia.org/api/rest_v1/page/summary/{requests.utils.quote(title)}",
        headers=headers,
        timeout=REQUEST_TIMEOUT,
    )
    summary_resp.raise_for_status()
    summary_data = summary_resp.json()
    extract = (summary_data.get("extract") or "").strip()
    if not extract:
        raise RuntimeError("Wikipedia returned empty summary")
    return extract


def call_offline_fallback(prompt: str) -> str:
    short = (prompt or "").strip()
    if len(short) > 160:
        short = short[:160] + "..."
    return (
        "OpenGradient gateway is temporarily unstable, so this response uses local fallback mode. "
        f"Your prompt was: \"{short}\". "
        "Try again in a minute for full TEE/x402 inference."
    )


def _github_headers() -> dict[str, str]:
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "OpenGradientRepoSecurityChecker/1.0",
    }
    github_token = (os.getenv("GITHUB_TOKEN") or "").strip()
    if github_token:
        headers["Authorization"] = f"Bearer {github_token}"
    return headers


def _parse_github_repo_input(repo_input: str) -> tuple[str, str, str]:
    value = (repo_input or "").strip()
    if not value:
        raise ValueError("GitHub repository URL is required")

    value = value.replace("git@github.com:", "https://github.com/").replace(".git", "")
    if re.match(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$", value):
        owner, repo = value.split("/", 1)
        return owner, repo, f"https://github.com/{owner}/{repo}"

    parsed = urlparse(value)
    if parsed.netloc.lower() not in ("github.com", "www.github.com"):
        raise ValueError("Only github.com repositories are supported")

    parts = [p for p in parsed.path.strip("/").split("/") if p]
    if len(parts) < 2:
        raise ValueError("Invalid GitHub repository URL. Expected format: https://github.com/owner/repo")

    owner, repo = parts[0], parts[1]
    return owner, repo, f"https://github.com/{owner}/{repo}"


def _fetch_repo_snapshot(owner: str, repo: str) -> dict[str, Any]:
    headers = _github_headers()
    repo_resp = requests.get(f"{GITHUB_API_BASE}/repos/{owner}/{repo}", headers=headers, timeout=REQUEST_TIMEOUT)
    if repo_resp.status_code == 404:
        raise RuntimeError("Repository not found or private repository is inaccessible")
    repo_resp.raise_for_status()
    repo_data = repo_resp.json()

    readme_text = ""
    readme_resp = requests.get(
        f"{GITHUB_API_BASE}/repos/{owner}/{repo}/readme",
        headers=headers,
        timeout=REQUEST_TIMEOUT,
    )
    if readme_resp.status_code == 200:
        readme_payload = readme_resp.json()
        content_b64 = (readme_payload.get("content") or "").replace("\n", "")
        if content_b64:
            try:
                readme_text = base64.b64decode(content_b64).decode("utf-8", errors="ignore")
            except Exception:
                readme_text = ""

    langs_resp = requests.get(
        f"{GITHUB_API_BASE}/repos/{owner}/{repo}/languages",
        headers=headers,
        timeout=REQUEST_TIMEOUT,
    )
    langs = {}
    if langs_resp.status_code == 200:
        try:
            langs = langs_resp.json()
        except Exception:
            langs = {}

    return {
        "repo": {
            "full_name": repo_data.get("full_name"),
            "html_url": repo_data.get("html_url"),
            "description": repo_data.get("description"),
            "default_branch": repo_data.get("default_branch"),
            "created_at": repo_data.get("created_at"),
            "updated_at": repo_data.get("updated_at"),
            "pushed_at": repo_data.get("pushed_at"),
            "stargazers_count": int(repo_data.get("stargazers_count") or 0),
            "forks_count": int(repo_data.get("forks_count") or 0),
            "open_issues_count": int(repo_data.get("open_issues_count") or 0),
            "watchers_count": int(repo_data.get("subscribers_count") or repo_data.get("watchers_count") or 0),
            "archived": bool(repo_data.get("archived")),
            "disabled": bool(repo_data.get("disabled")),
            "license": (repo_data.get("license") or {}).get("spdx_id"),
            "topics": repo_data.get("topics") or [],
        },
        "languages": langs,
        "readme_excerpt": (readme_text or "")[:REPO_CHECK_README_MAX_CHARS],
    }


def _safe_parse_iso8601(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)
    except Exception:
        return None


def _build_repo_heuristics(snapshot: dict[str, Any]) -> dict[str, Any]:
    repo = snapshot.get("repo") or {}
    readme = (snapshot.get("readme_excerpt") or "").lower()
    risk_score = 0
    findings: list[str] = []

    if repo.get("archived"):
        risk_score += 30
        findings.append("Repository is archived (maintenance risk).")
    if repo.get("disabled"):
        risk_score += 40
        findings.append("Repository is disabled by GitHub.")
    if not repo.get("license") or repo.get("license") in ("NOASSERTION", "NONE", None):
        risk_score += 10
        findings.append("No clear open-source license detected.")
    if not (snapshot.get("readme_excerpt") or "").strip():
        risk_score += 15
        findings.append("README is missing or inaccessible.")

    created_at = _safe_parse_iso8601(repo.get("created_at"))
    pushed_at = _safe_parse_iso8601(repo.get("pushed_at"))
    now_utc = datetime.now(timezone.utc)
    stars = int(repo.get("stargazers_count") or 0)

    if created_at and (now_utc - created_at).days < 14 and stars < 3:
        risk_score += 15
        findings.append("Very new repository with low social proof.")
    if pushed_at and (now_utc - pushed_at).days > 365:
        risk_score += 10
        findings.append("No recent commits in over 1 year.")
    if int(repo.get("open_issues_count") or 0) > 200:
        risk_score += 5
        findings.append("Large number of open issues.")

    suspicious_patterns = {
        "curl pipe shell install detected": r"curl[^\\n]*\|[^\\n]*(sh|bash)",
        "wget pipe shell install detected": r"wget[^\\n]*\|[^\\n]*(sh|bash)",
        "private key string mention in README": r"private[_ -]?key|mnemonic|seed phrase",
        "disabled ssl verification mention": r"verify\s*=\s*false|--insecure|ssl\s*verify\s*false",
    }
    for label, pattern in suspicious_patterns.items():
        if re.search(pattern, readme):
            risk_score += 10
            findings.append(label)

    risk_score = max(0, min(risk_score, 100))
    if risk_score >= 75:
        verdict = "critical"
    elif risk_score >= 50:
        verdict = "high"
    elif risk_score >= 25:
        verdict = "medium"
    else:
        verdict = "low"

    return {
        "risk_score": risk_score,
        "verdict": verdict,
        "findings": findings,
    }


def _build_repo_security_prompt(snapshot: dict[str, Any], heuristics: dict[str, Any], focus: str) -> str:
    return (
        "You are a senior Web3 + AppSec auditor. Analyze the GitHub repository security posture.\n"
        f"Focus area from user: {focus or 'general repository security'}.\n\n"
        "Repository metadata:\n"
        f"{json.dumps(snapshot.get('repo') or {}, ensure_ascii=False, indent=2)}\n\n"
        "Languages:\n"
        f"{json.dumps(snapshot.get('languages') or {}, ensure_ascii=False, indent=2)}\n\n"
        "Heuristic pre-scan:\n"
        f"{json.dumps(heuristics, ensure_ascii=False, indent=2)}\n\n"
        "README excerpt:\n"
        f"{(snapshot.get('readme_excerpt') or '')[:3500]}\n\n"
        "Return concise markdown with sections:\n"
        "1) Verdict (Low/Medium/High/Critical and why)\n"
        "2) Top 5 risks (specific)\n"
        "3) What to verify manually next\n"
        "4) Safe usage recommendation for end users\n"
        "Use concrete language, no legal advice."
    )


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
            "has_og_alpha_private_key": bool(os.getenv("OG_ALPHA_PRIVATE_KEY")),
            "alpha_key_source": "OG_ALPHA_PRIVATE_KEY" if os.getenv("OG_ALPHA_PRIVATE_KEY") else ("OG_PRIVATE_KEY" if os.getenv("OG_PRIVATE_KEY") else "none"),
            "has_alpha_private_key": bool(os.getenv("OG_ALPHA_PRIVATE_KEY") or os.getenv("OG_PRIVATE_KEY")),
            "og_sdk_available": og is not None,
            "og_sdk_model": OG_SDK_MODEL,
            "og_settlement_mode": OG_SETTLEMENT_MODE,
            "model_hub_configured": bool(os.getenv("OG_HUB_EMAIL") and os.getenv("OG_HUB_PASSWORD")),
            "x402_endpoint": X402_ENDPOINT,
            "x402_fallback_endpoints": X402_FALLBACK_ENDPOINTS,
            "x402_client_sync_available": X402ClientSync is not None,
            "x402_exact_evm_scheme_available": X402ExactEvmClientScheme is not None,
            "x402_header_encoder_available": encode_payment_signature_header is not None,
            "x402_decode_required_available": decode_payment_required_header is not None,
            "x402_import_errors": X402_IMPORT_ERRORS[:3],
            "wiki_fallback_enabled": ENABLE_WIKI_FALLBACK,
            "build_marker": BUILD_MARKER,
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


@app.post("/api/repo-security/check")
def repo_security_check():
    data = request.get_json(silent=True) or {}
    repo_input = (data.get("repo_url") or data.get("repo") or "").strip()
    focus = (data.get("focus") or "smart-contract and dependency risk").strip()

    try:
        owner, repo, canonical_url = _parse_github_repo_input(repo_input)
    except ValueError as exc:
        return _json_error(str(exc), 400)

    try:
        snapshot = _fetch_repo_snapshot(owner, repo)
        heuristics = _build_repo_heuristics(snapshot)
        analysis_prompt = _build_repo_security_prompt(snapshot, heuristics, focus)
        analysis, ai_provider = generate_reply(analysis_prompt)
        return jsonify(
            {
                "ok": True,
                "repo": canonical_url,
                "metadata": snapshot.get("repo") or {},
                "languages": snapshot.get("languages") or {},
                "heuristics": heuristics,
                "analysis": analysis,
                "provider": ai_provider,
            }
        )
    except requests.HTTPError as exc:
        details = exc.response.text[:500] if exc.response is not None else str(exc)
        return _json_error("GitHub API error", 502, details)
    except requests.RequestException as exc:
        return _json_error("Network error while fetching repository", 502, str(exc))
    except Exception as exc:
        return _json_error("Repo security check failed", 500, str(exc))


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

        auto_payment_signature = None
        auto_sign_error = None
        x402_headers = _extract_x402_headers(response.headers)
        if response.status_code == 402:
            payment_required_header = _get_header_case_insensitive(x402_headers, PAYMENT_REQUIRED_HEADER)
            if payment_required_header and os.getenv("OG_PRIVATE_KEY"):
                try:
                    auto_payment_signature = _sign_payment_required_header(payment_required_header)
                except Exception as exc:
                    auto_sign_error = str(exc)

        return jsonify(
            {
                "ok": response.status_code in (200, 402),
                "status_code": response.status_code,
                "endpoint_used": endpoint_used,
                "headers": x402_headers,
                "body": body,
                "auto_payment_signature": auto_payment_signature,
                "auto_sign_error": auto_sign_error,
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
    payment_signature = (data.get("payment_signature") or "").strip()
    messages = data.get("messages")

    if not x_payment and not payment_signature:
        return _json_error("x_payment or payment_signature is required", 400)
    if not isinstance(messages, list) or not messages:
        return _json_error("messages must be a non-empty array", 400)

    headers = {
        "Content-Type": "application/json",
        "X-SETTLEMENT-TYPE": settlement,
    }
    if payment_signature:
        headers["PAYMENT-SIGNATURE"] = payment_signature
    if x_payment:
        headers["X-PAYMENT"] = x_payment
        if not payment_signature:
            headers["PAYMENT-SIGNATURE"] = x_payment

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
        return _json_error("Alpha inference failed", 500, _build_alpha_error_details(exc, mode, model_cid))


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









