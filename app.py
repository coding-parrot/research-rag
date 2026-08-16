"""The chatbot: one POST endpoint and a chat page.

Run:  .venv/bin/uvicorn app:app --port 8477
Then open http://127.0.0.1:8477
"""

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from rag.pipeline import ask

app = FastAPI(title="research-rag")


class Question(BaseModel):
    question: str


@app.post("/ask")
def post_ask(body: Question) -> dict:
    answer = ask(body.question)
    return {
        "answer": answer.answer,
        "citations": answer.citations,
        "refused": answer.refused,
    }


PAGE = """<!doctype html>
<meta charset="utf-8"><title>research-rag</title>
<style>
  body { background:#111; color:#eee; font:16px/1.6 system-ui; max-width:52rem; margin:3rem auto; padding:0 1rem }
  input { flex:1; padding:.7rem; font-size:1rem; background:#1c2128; color:#eee; border:1px solid #333; border-radius:8px }
  button { padding:.7rem 1.4rem; font-size:1rem; background:#4a6cd4; color:#fff; border:0; border-radius:8px; cursor:pointer }
  form { display:flex; gap:.6rem }
  #out { margin-top:1.5rem; white-space:pre-wrap }
  .src { color:#8b949e; font-size:.9rem; margin-top:1rem }
  .refused { color:#e3b341 }
</style>
<h2>research-rag</h2>
<p>Ask about the indexed papers. Answers cite their sources.</p>
<form id="f"><input id="q" placeholder="How does LoRA reduce trainable parameters?" autofocus>
<button>Ask</button></form>
<div id="out"></div>
<script>
f.onsubmit = async (e) => {
  e.preventDefault();
  out.textContent = "thinking...";
  const r = await fetch("/ask", {method:"POST", headers:{"content-type":"application/json"},
                                 body: JSON.stringify({question: q.value})});
  const d = await r.json();
  out.textContent = "";
  const p = document.createElement("p");
  p.textContent = d.answer;
  if (d.refused) p.className = "refused";
  out.append(p);
  d.citations.forEach((c, i) => {
    const s = document.createElement("div");
    s.className = "src";
    s.textContent = `[${i+1}] ${c.source} - "${c.quote}"`;
    out.append(s);
  });
};
</script>"""


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    return PAGE


if __name__ == "__main__":
    # So "Run app.py" works directly in an IDE - no terminal needed.
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8477)
