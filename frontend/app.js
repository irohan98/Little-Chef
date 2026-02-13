const statusEl = (id, message, isError = false) => {
  const el = document.getElementById(id);
  if (!el) return;
  el.textContent = message;
  el.style.color = isError ? "#b42318" : "#2b4d35";
};

const refreshStats = async () => {
  const el = document.getElementById("recipeCount");
  if (!el) return;
  try {
    const res = await fetch("/stats");
    if (!res.ok) throw new Error("stats failed");
    const data = await res.json();
    el.textContent = data.recipes ?? "0";
  } catch (_) {
    el.textContent = "—";
  }
};

const scrollButtons = document.querySelectorAll("[data-scroll]");
scrollButtons.forEach((btn) => {
  btn.addEventListener("click", () => {
    const target = document.getElementById(btn.dataset.scroll);
    if (target) target.scrollIntoView({ behavior: "smooth" });
  });
});

const saveTextBtn = document.getElementById("saveText");
if (saveTextBtn) {
  saveTextBtn.addEventListener("click", async () => {
    const text = document.getElementById("recipeText").value.trim();
    if (!text) {
      statusEl("textStatus", "Paste a recipe first.", true);
      return;
    }

    statusEl("textStatus", "Saving...");
    const res = await fetch("/ingest/text", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });

    if (!res.ok) {
      const err = await res.json();
      statusEl("textStatus", err.detail || "Failed to store recipe.", true);
      return;
    }
    statusEl("textStatus", "Recipe saved.");
    refreshStats();
  });
}

const saveYoutubeBtn = document.getElementById("saveYoutube");
if (saveYoutubeBtn) {
  saveYoutubeBtn.addEventListener("click", async () => {
    const link = document.getElementById("youtubeLink").value.trim();
    const cookies_path = document.getElementById("cookiesPath").value.trim();
    if (!link) {
      statusEl("ytStatus", "Add a YouTube link first.", true);
      return;
    }

    statusEl("ytStatus", "Ingesting... this can take a minute.");
    const res = await fetch("/ingest/youtube", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ link, cookies_path }),
    });

    if (!res.ok) {
      const err = await res.json();
      statusEl("ytStatus", err.detail || "Failed to ingest YouTube recipe.", true);
      return;
    }

    statusEl("ytStatus", "YouTube recipe stored.");
    refreshStats();
  });
}

const searchBtn = document.getElementById("searchBtn");
if (searchBtn) {
  searchBtn.addEventListener("click", async () => {
    const query = document.getElementById("query").value.trim();
    const top_k = Number(document.getElementById("topK").value || 3);
    if (!query) return;

    const res = await fetch("/search", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, top_k }),
    });

    const resultsEl = document.getElementById("results");
    resultsEl.innerHTML = "";

    if (!res.ok) {
      const err = await res.json();
      resultsEl.innerHTML = `<div class="status">${err.detail || "Search failed."}</div>`;
      return;
    }

    const data = await res.json();
    if (!data.results.length) {
      resultsEl.innerHTML = "<div class=\"status\">No matches found.</div>";
      return;
    }

    data.results.forEach((item, idx) => {
      const card = document.createElement("div");
      card.className = "result-card";
      card.innerHTML = `
        <h4>${idx + 1}. ${item.title}</h4>
        <div>Distance: ${item.distance?.toFixed(4) ?? "n/a"}</div>
        <p>${item.text}</p>
      `;
      resultsEl.appendChild(card);
    });
  });
}

const ragBtn = document.getElementById("ragBtn");
if (ragBtn) {
  ragBtn.addEventListener("click", async () => {
    const query = document.getElementById("ragQuery").value.trim();
    const top_k = Number(document.getElementById("ragTopK").value || 3);
    const api_key = document.getElementById("ragKey").value.trim();

    if (!query) return;

    const output = document.getElementById("ragOutput");
    output.textContent = "Generating...";

    const res = await fetch("/rag", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, top_k, api_key: api_key || null }),
    });

    if (!res.ok) {
      const err = await res.json();
      output.textContent = err.detail || "RAG failed.";
      return;
    }

    const data = await res.json();
    output.textContent = data.response || "No response.";
  });
}

refreshStats();
