"""Streamlit UI for the FinSolve RBAC RAG chatbot."""

from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Any, Iterator

import requests
import streamlit as st
from supabase import Client, create_client


DEFAULT_BACKEND_URL = "http://127.0.0.1:8000"
ADMIN_ROLES = {"admin", "c-level"}
AVAILABLE_ROLES = ["employee", "finance", "hr", "marketing", "engineering", "c-level", "admin"]


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
    backend_url = os.getenv("BACKEND_URL", DEFAULT_BACKEND_URL).strip().rstrip("/")
    if backend_url and not backend_url.startswith(("http://", "https://")):
        backend_url = f"https://{backend_url}"
    return backend_url


def inject_custom_css() -> None:
    st.markdown(
        """
        <style>
            #MainMenu,
            footer,
            [data-testid="stToolbar"],
            [data-testid="stDecoration"],
            [data-testid="stDeployButton"] {
                display: none !important;
                visibility: hidden !important;
            }

            [data-testid="stChatMessage"] {
                border-radius: 10px;
                padding: 0.35rem 0.45rem;
            }

            [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
                background: rgba(37, 99, 235, 0.10);
                border: 1px solid rgba(37, 99, 235, 0.35);
            }

            [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
                background: rgba(30, 41, 59, 0.72);
                border: 1px solid rgba(148, 163, 184, 0.16);
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


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


def extract_user_role(user: dict[str, Any]) -> str | None:
    metadata = user.get("user_metadata")
    if not isinstance(metadata, dict):
        metadata = user.get("raw_user_meta_data")
    if isinstance(metadata, dict):
        role = metadata.get("role")
        if isinstance(role, str) and role.strip():
            return role.strip()
    return None


def initialize_session_state() -> None:
    st.session_state.setdefault("access_token", None)
    st.session_state.setdefault("refresh_token", None)
    st.session_state.setdefault("user_email", None)
    st.session_state.setdefault("user_role", None)
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("session_id", None)


def login(email: str, password: str) -> None:
    response = get_supabase_client().auth.sign_in_with_password(
        {
            "email": email,
            "password": password,
        }
    )
    session = extract_session(response)
    access_token = session.get("access_token")
    refresh_token = session.get("refresh_token")

    if not access_token:
        raise RuntimeError("Login succeeded but no access token was returned.")

    user = extract_user(response)
    st.session_state.access_token = access_token
    st.session_state.refresh_token = refresh_token
    st.session_state.user_email = user.get("email", email)
    st.session_state.user_role = extract_user_role(user)
    st.session_state.messages = []
    st.session_state.session_id = None


def logout() -> None:
    st.session_state.access_token = None
    st.session_state.refresh_token = None
    st.session_state.user_email = None
    st.session_state.user_role = None
    st.session_state.messages = []
    st.session_state.session_id = None


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


def stream_chat_api(question: str) -> Iterator[dict[str, Any]]:
    def response_detail(response: requests.Response) -> str:
        detail = response.text
        try:
            detail = response.json().get("detail", detail)
        except ValueError:
            pass
        return str(detail)

    def post_chat() -> requests.Response:
        payload: dict[str, Any] = {
            "question": question,
            "top_k": 5,
        }
        if st.session_state.session_id:
            payload["session_id"] = st.session_state.session_id

        return requests.post(
            f"{get_backend_url()}/api/chat",
            json=payload,
            headers={
                "Authorization": f"Bearer {st.session_state.access_token}",
            },
            stream=True,
            timeout=120,
        )

    response = post_chat()

    if response.status_code == 401:
        response.close()
        if refresh_session():
            response = post_chat()

        if response.status_code == 401:
            detail = response_detail(response)
            response.close()
            logout()
            raise RuntimeError(
                "Authentication failed. Please sign in again. "
                f"Backend detail: {detail}"
            )

    if response.status_code >= 400:
        detail = response_detail(response)
        response.close()
        raise RuntimeError(detail)

    try:
        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError("Received malformed chat stream data.") from exc
    finally:
        response.close()


def api_headers() -> dict[str, str]:
    return {
        "Authorization": f"Bearer {st.session_state.access_token}",
        "Content-Type": "application/json",
    }


def backend_request(method: str, path: str, **kwargs: Any) -> requests.Response:
    response = requests.request(
        method,
        f"{get_backend_url()}{path}",
        headers=api_headers(),
        timeout=30,
        **kwargs,
    )

    if response.status_code == 401 and refresh_session():
        response = requests.request(
            method,
            f"{get_backend_url()}{path}",
            headers=api_headers(),
            timeout=30,
            **kwargs,
        )

    return response


def fetch_admin_users() -> list[dict[str, Any]]:
    response = backend_request("GET", "/api/admin/users")
    if response.status_code >= 400:
        raise RuntimeError(response.text)
    payload = response.json()
    users = payload.get("users", [])
    return users if isinstance(users, list) else []


def update_user_role(user_id: str, new_role: str) -> None:
    response = backend_request(
        "POST",
        "/api/admin/update_role",
        json={
            "user_id": user_id,
            "new_role": new_role,
        },
    )
    if response.status_code >= 400:
        raise RuntimeError(response.text)


def refresh_session() -> bool:
    refresh_token = st.session_state.get("refresh_token")
    if not refresh_token:
        return False

    try:
        response = get_supabase_client().auth.refresh_session(refresh_token)
        session = extract_session(response)
    except Exception:
        return False

    access_token = session.get("access_token")
    if not access_token:
        return False

    st.session_state.access_token = access_token
    st.session_state.refresh_token = session.get("refresh_token") or refresh_token
    return True


def get_supabase_rest_headers() -> dict[str, str]:
    return {
        "apikey": require_env("SUPABASE_KEY"),
        "Authorization": f"Bearer {st.session_state.access_token}",
        "Content-Type": "application/json",
    }


def supabase_rest_get(path: str, params: dict[str, str]) -> list[dict[str, Any]]:
    response = requests.get(
        f"{require_env('SUPABASE_URL').rstrip('/')}/rest/v1/{path}",
        headers=get_supabase_rest_headers(),
        params=params,
        timeout=30,
    )

    if response.status_code == 401 and refresh_session():
        response = requests.get(
            f"{require_env('SUPABASE_URL').rstrip('/')}/rest/v1/{path}",
            headers=get_supabase_rest_headers(),
            params=params,
            timeout=30,
        )

    if response.status_code >= 400:
        raise RuntimeError(response.text)

    data = response.json()
    return data if isinstance(data, list) else []


def fetch_chat_sessions() -> list[dict[str, Any]]:
    return supabase_rest_get(
        "chat_sessions",
        {
            "select": "id,title,created_at",
            "order": "created_at.desc",
        },
    )


def fetch_chat_messages(session_id: str) -> list[dict[str, Any]]:
    return supabase_rest_get(
        "chat_messages",
        {
            "select": "role,content,created_at",
            "session_id": f"eq.{session_id}",
            "order": "created_at.asc",
        },
    )


def load_chat_session(session_id: str) -> None:
    rows = fetch_chat_messages(session_id)
    messages: list[dict[str, Any]] = []
    for row in rows:
        role = row.get("role")
        content = row.get("content")
        if role in {"user", "assistant"} and isinstance(content, str):
            messages.append(
                {
                    "role": role,
                    "content": content,
                    "sources": [],
                }
            )

    st.session_state.session_id = session_id
    st.session_state.messages = messages


def start_new_chat() -> None:
    st.session_state.messages = []
    st.session_state.session_id = None


def render_admin_panel() -> None:
    if (st.session_state.user_role or "").strip().lower() not in ADMIN_ROLES:
        return

    st.divider()
    st.caption("Admin")

    with st.expander("User Roles", expanded=False):
        try:
            users = fetch_admin_users()
        except Exception as exc:
            st.caption(f"Unable to load users: {get_safe_error_message(exc)}")
            return

        if not users:
            st.caption("No users found.")
            return

        for user in users:
            user_id = user.get("id")
            email = user.get("email") or "Unknown user"
            current_role = user.get("role") or "employee"
            if not isinstance(user_id, str) or not user_id:
                continue

            st.caption(str(email))
            default_index = (
                AVAILABLE_ROLES.index(current_role)
                if current_role in AVAILABLE_ROLES
                else 0
            )
            selected_role = st.selectbox(
                "Role",
                AVAILABLE_ROLES,
                index=default_index,
                key=f"role_select_{user_id}",
                label_visibility="collapsed",
            )

            if st.button("Update Role", key=f"role_update_{user_id}", use_container_width=True):
                try:
                    update_user_role(user_id, selected_role)
                    st.success(f"Updated {email} to {selected_role}.")
                except Exception as exc:
                    st.error(f"Unable to update role: {get_safe_error_message(exc)}")


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
        if st.session_state.user_role:
            st.caption(f"Role: {st.session_state.user_role}")

        if st.button("New Chat", use_container_width=True):
            start_new_chat()
            st.rerun()

        st.divider()
        st.caption("Past chats")
        try:
            sessions = fetch_chat_sessions()
        except Exception as exc:
            sessions = []
            st.caption(f"Unable to load chats: {get_safe_error_message(exc)}")

        if not sessions:
            st.caption("No saved chats yet.")

        for session in sessions:
            session_id = session.get("id")
            if not isinstance(session_id, str):
                continue

            title = session.get("title")
            if not isinstance(title, str) or not title.strip():
                title = "Untitled chat"

            is_active = session_id == st.session_state.session_id
            label = f"• {title}" if is_active else title
            if st.button(label, key=f"session_{session_id}", use_container_width=True):
                try:
                    load_chat_session(session_id)
                    st.rerun()
                except Exception as exc:
                    st.error(f"Unable to load chat: {get_safe_error_message(exc)}")

        st.divider()
        render_admin_panel()
        st.divider()
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
        sources: list[dict[str, Any]] = []
        streamed_parts: list[str] = []
        session_id = st.session_state.session_id

        def token_stream() -> Iterator[str]:
            nonlocal session_id, sources

            for event in stream_chat_api(prompt):
                event_type = event.get("type")
                if event_type == "token":
                    token = str(event.get("content", ""))
                    streamed_parts.append(token)
                    yield token
                elif event_type == "sources":
                    source_payload = event.get("sources", [])
                    sources = source_payload if isinstance(source_payload, list) else []
                    if isinstance(event.get("session_id"), str):
                        session_id = event["session_id"]
                elif event_type == "error":
                    raise RuntimeError(event.get("message", "Chat stream failed."))

        try:
            answer = st.write_stream(token_stream)
            if not answer:
                answer = "I could not generate a response. Please try again."
                st.markdown(answer)
        except Exception as exc:
            answer = "".join(streamed_parts) or str(exc)
            sources = []
            st.markdown(answer)

        render_sources(sources)
        st.session_state.session_id = session_id

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
    inject_custom_css()
    initialize_session_state()

    if st.session_state.access_token:
        render_chat()
    else:
        render_login()


if __name__ == "__main__":
    main()
