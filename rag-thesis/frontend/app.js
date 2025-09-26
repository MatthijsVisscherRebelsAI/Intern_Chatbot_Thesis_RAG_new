const API = location.origin; // same-origin backend
const q = document.getElementById("q");
const btn = document.getElementById("send");
const log = document.getElementById("log");
const faqButtons = document.querySelectorAll("button.faq");

// Ensure only one active EventSource at a time
let activeES = null;
function setStreamingUI(isStreaming) {
  btn.disabled = isStreaming;
  q.disabled = isStreaming;
  btn.textContent = isStreaming ? "Streaming..." : "Ask";
}
window.addEventListener("beforeunload", () => {
  try { activeES && activeES.close(); } catch {}
});

async function ask() {
  // Guard: avoid opening multiple streams
  if (activeES) return;
  const question = q.value.trim();
  if (!question) return;
  q.value = "";
  append("You", question);
  append("Bot", "");
  setStreamingUI(true);

  try {
    const url = new URL(`${API}/stream`);
    url.searchParams.set("question", question);
    
    const es = new EventSource(url.toString());
    activeES = es;

    let buffer = "";
    const flush = () => replaceLast("Bot", buffer);
    let streamStarted = false; // becomes true after first chunk

    // If the EventSource reconnects after streaming started, close the reconnection to avoid duplicates
    es.addEventListener("open", () => {
      if (streamStarted) {
        try { es.close(); } catch {}
      }
    });

    es.addEventListener("chunk", (e) => {
      try {
        const { delta } = JSON.parse(e.data);
        if (typeof delta === "string") {
          // Handle both cumulative and incremental deltas robustly
          if (delta.length >= buffer.length && delta.startsWith(buffer)) {
            buffer = delta; // cumulative update
          } else if (!buffer.endsWith(delta)) {
            buffer += delta; // incremental token
          }
          flush();
          streamStarted = true;
        }
      } catch {}
    });

    es.addEventListener("done", (e) => {
      try {
        const payload = JSON.parse(e.data || '{}');
        const cites = Array.isArray(payload.citations) ? payload.citations : [];
        // Update Bot message with a plain-text sources summary by unique page
        if (cites.length) {
          // Build first-occurence page -> url map
          const pageToUrl = new Map();
          for (const c of cites) {
            if (c.page !== undefined && c.page !== null && !pageToUrl.has(c.page)) {
              pageToUrl.set(c.page, c.url || null);
            }
          }
          const pages = Array.from(pageToUrl.keys()).sort((a, b) => a - b);
          if (pages.length) {
            const formattedLines = pages.map((p) => {
              const url = pageToUrl.get(p);
              const label = `p.${p}`;
              return url ? `<a href="${url}" target="_blank" rel="noopener noreferrer">${label}</a>` : label;
            }).join('\n');
            buffer += `\n\nSources:\n${formattedLines}`;
            flush();
          } else {
            buffer += `\n\nSources: None`;
            flush();
          }
        } else {
          buffer += `\n\nSources: None`;
          flush();
        }
        // Remove separate sources box; links are now under the Bot message
      } catch {}
      try { es.close(); } catch {}
      activeES = null;
      setStreamingUI(false);
    });

    es.onerror = () => {
      try { es.close(); } catch {}
      activeES = null;
      setStreamingUI(false);
    };
  } catch (e) {
    console.error(e);
    activeES = null;
    setStreamingUI(false);
  }
}

function append(who, text) {
  const div = document.createElement("div");
  div.className = "msg";
  div.innerHTML = `<small>${who}</small>\n${text}`;
  log.appendChild(div);
}

function replaceLast(who, text) {
  const nodes = [...log.querySelectorAll(".msg")];
  const last = nodes[nodes.length - 1];
  last.innerHTML = `<small>${who}</small>\n${text}`;
}

btn.addEventListener("click", ask);
q.addEventListener("keydown", (e) => e.key === "Enter" && ask());

faqButtons.forEach((b) => {
  b.addEventListener("click", () => {
    q.value = b.dataset.q || b.textContent.trim();
    ask();
  });
});
