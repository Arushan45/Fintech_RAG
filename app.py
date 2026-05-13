"""Streamlit UI for the FinSolve RBAC RAG chatbot."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import requests
import streamlit as st
from supabase import Client, create_client


DEFAULT_BACKEND_URL = "http://127.0.0.1:8000"


def load_env_file(path: Path = Path(".env")) -> None:
    """Load simple KEY=VALUE pairs from .env without overriding real env vars."""

    if not path.exists():
        return

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")

        if key and key not in os.environ:
            os.environ[key] = value


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"{name} must be configured")
    return value


@st.cache_resource
def get_supabase_client() -> Client:
    return create_client(
        require_env("SUPABASE_URL"),
        require_env("SUPABASE_KEY"),
    )


def get_backend_url() -> str:
    return os.getenv("BACKEND_URL", DEFAULT_BACKEND_URL).rstrip("/")


def as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "dict"):
        return value.dict()
    return {}


def extract_session(sign_in_response: Any) -> dict[str, Any]:
    response = as_dict(sign_in_response)
    if isinstance(response.get("session"), dict):
        return response["session"]
    if isinstance(response.get("data"), dict) and isinstance(
        response["data"].get("session"),
        dict,
    ):
        return response["data"]["session"]

    session = getattr(sign_in_response, "session", None)
    if session is not None:
        return as_dict(session)

    data = getattr(sign_in_response, "data", None)
    if data is not None:
        data_dict = as_dict(data)
        if isinstance(data_dict.get("session"), dict):
            return data_dict["session"]

    return {}


def extract_user(sign_in_response: Any) -> dict[str, Any]:
    response = as_dict(sign_in_response)
    if isinstance(response.get("user"), dict):
        return response["user"]
    if isinstance(response.get("data"), dict) and isinstance(
        response["data"].get("user"),
        dict,
    ):
        return response["data"]["user"]

    user = getattr(sign_in_response, "user", None)
    if user is not None:
        return as_dict(user)

    data = getattr(sign_in_response, "data", None)
    if data is not None:
        data_dict = as_dict(data)
        if isinstance(data_dict.get("user"), dict):
            return data_dict["user"]

    return {}


def get_safe_error_message(exc: Exception) -> str:
    message = getattr(exc, "message", None) or str(exc)
    return message or "Check your email, password, and Supabase configuration."


def initialize_session_state() -> None:
    st.session_state.setdefault("access_token", None)
    st.session_state.setdefault("user_email", None)
    st.session_state.setdefault("messages", [])


def login(email: str, password: str) -> None:
    response = get_supabase_client().auth.sign_in_with_password(
        {
            "email": email,
            "password": password,
        }
    )
    session = extract_session(response)
    access_token = session.get("access_token")

    if not access_token:
        raise RuntimeError("Login succeeded but no access token was returned.")

    user = extract_user(response)
    st.session_state.access_token = access_token
    st.session_state.user_email = user.get("email", email)
    st.session_state.messages = []


def logout() -> None:
    st.session_state.access_token = None
    st.session_state.user_email = None
    st.session_state.messages = []


def render_login() -> None:
    st.title("FinSolve AI")
    st.subheader("Secure Knowledge Assistant")

    with st.form("login_form", clear_on_submit=False):
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Sign in", type="primary")

    if not submitted:
        return

    if not email or not password:
        st.error("Enter both email and password.")
        return

    try:
        login(email=email, password=password)
        st.rerun()
    except Exception as exc:
        st.error(f"Unable to sign in: {get_safe_error_message(exc)}")


def call_chat_api(question: str) -> dict[str, Any]:
    response = requests.post(
        f"{get_backend_url()}/api/chat",
        json={
            "question": question,
            "top_k": 5,
        },
        headers={
            "Authorization": f"Bearer {st.session_state.access_token}",
        },
        timeout=120,
    )

    if response.status_code == 401:
        logout()
        raise RuntimeError("Your session expired. Please sign in again.")

    if response.status_code >= 400:
        detail = response.text
        try:
            detail = response.json().get("detail", detail)
        except ValueError:
            pass
        raise RuntimeError(str(detail))

    return response.json()


def render_sources(sources: list[dict[str, Any]]) -> None:
    with st.expander("Source References"):
        if not sources:
            st.caption("No source chunks were returned.")
            return

        for index, source in enumerate(sources, start=1):
            metadata = source.get("metadata", {})
            source_path = metadata.get("source", "unknown")
            access_level = metadata.get("access_level", "unknown")
            chunk_index = metadata.get("chunk_index", "unknown")

            st.markdown(
                f"**Source {index}**  \n"
                f"`{source_path}`  \n"
                f"Access: `{access_level}` | Chunk: `{chunk_index}`"
            )
            st.code(source.get("content", ""), language="markdown")


def render_chat() -> None:
    st.title("FinSolve AI")

    with st.sidebar:
        st.caption("Signed in as")
        st.write(st.session_state.user_email)
        if st.button("Sign out"):
            logout()
            st.rerun()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                render_sources(message.get("sources", []))

    prompt = st.chat_input("Ask a question about FinSolve")
    if not prompt:
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Checking your permitted knowledge base..."):
            try:
                payload = call_chat_api(prompt)
                answer = payload.get("answer", "")
                sources = payload.get("sources", [])
            except Exception as exc:
                answer = str(exc)
                sources = []

        st.markdown(answer)
        render_sources(sources)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources,
        }
    )


def main() -> None:
    load_env_file()
    st.set_page_config(page_title="FinSolve AI", page_icon="FS", layout="centered")
    initialize_session_state()

    if st.session_state.access_token:
        render_chat()
    else:
        render_login()


if __name__ == "__main__":
    main()
