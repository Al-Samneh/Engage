const chatLogEl = document.getElementById("chatLog");
const chatForm = document.getElementById("chatForm");
const chatInput = document.getElementById("chatInput");
const chatStatusEl = document.getElementById("chatStatus");
const filterBoxEl = document.getElementById("filterBox");
const conversationState = { current: null };

function appendMessage(role, text) {
  const wrap = document.createElement("div");
  wrap.className = `chat-message ${role === "user" ? "user" : "ai"}`;
  wrap.innerHTML = `<div class="label">${role === "user" ? "You" : "AI"}</div><div class="bubble">${text}</div>`;
  chatLogEl.appendChild(wrap);
  chatLogEl.scrollTop = chatLogEl.scrollHeight;
}

function renderFilters(filters) {
  filterBoxEl.textContent = JSON.stringify(filters || {}, null, 2);
}

async function handleChatSubmit(ev) {
  ev.preventDefault();
  const question = chatInput.value.trim();
  if (!question) return;
  appendMessage("user", question);
  chatInput.value = "";

  const conversationId =
    conversationState.current || (conversationState.current = crypto.randomUUID());

  const payload = {
    question,
    conversation_id: conversationId,
    metadata: {
      channel: "web-demo",
    },
  };

  chatStatusEl.textContent = "Thinking...";
  chatStatusEl.className = "status";

  try {
    const response = await fetch("/v1/restaurants/search", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!response.ok) throw new Error(`API error ${response.status}`);
    const data = await response.json();
    const answer = data.data.answer || "No answer returned.";
    appendMessage("ai", answer);
    renderFilters(data.data.applied_filters);
    // Supporting documents hidden per updated UI.
    chatStatusEl.textContent = `Trace ID: ${data.trace_id} · Latency: ${data.latency_ms}ms`;
    chatStatusEl.className = "status success";
  } catch (err) {
    appendMessage("ai", `Error: ${err.message}`);
    chatStatusEl.textContent = `Request failed: ${err.message}`;
    chatStatusEl.className = "status error";
  }
}

chatForm.addEventListener("submit", handleChatSubmit);

