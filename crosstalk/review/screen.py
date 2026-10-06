"""Self-contained local review screen. No server, vendor calls, or external assets."""

import json
from pathlib import Path

from crosstalk.review.workflow import snapshot

PAGE = r"""<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Crosstalk · Episode review</title>
<style>
:root{color-scheme:dark;--bg:#11151b;--card:#1a2029;--muted:#a8b6c7;--line:#334052;--accent:#b9da94}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:#f1f4f8;font:16px/1.6 system-ui}header,main{max-width:1060px;margin:auto;padding:28px}h1{font-size:32px;margin:4px 0}h2{font-size:21px}p{margin:10px 0}.muted,small{color:var(--muted)}.tag{color:var(--accent);text-transform:uppercase;font-size:12px;letter-spacing:2px}audio{width:100%;margin-top:12px}.toolbar{position:sticky;top:0;background:var(--bg);border-bottom:1px solid var(--line);padding:14px 0;display:flex;gap:12px;flex-wrap:wrap;z-index:1}input,select,textarea,button{font:inherit;color:inherit;background:#11151b;border:1px solid var(--line);border-radius:7px;padding:10px}input[type=search]{flex:1;min-width:180px}button{background:var(--accent);color:#192217;cursor:pointer;font-weight:650}article{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:24px;margin:20px 0}blockquote{margin:14px 0;padding-left:18px;border-left:3px solid var(--accent)}summary{cursor:pointer;color:var(--accent)}a{color:#acd7ff}textarea{display:block;width:100%;min-height:90px;margin:7px 0 16px}label{display:block;margin:12px 0}label.passage{border-top:1px solid var(--line);padding:12px 0}.passage p{white-space:pre-wrap;font-size:14px}.decision{width:100%}.empty{padding:30px;text-align:center}.notice{border-left:3px solid #d9bd80;padding-left:14px}.counts{font-variant-numeric:tabular-nums}
</style>
<header><div class="tag">Crosstalk · Local editorial desk</div><h1 id="topic">Episode review</h1>
<p class="muted">Review the exact spoken claims beside their sources. Keep policy disagreements distinct from factual support.</p>
<audio controls preload="metadata" src="../audio/episode_preview.mp3"></audio>
<p class="notice">Decisions stay in this browser until you download and import them. Proposed corrections are saved for review; they do not alter the transcript. Review completion does not publish or approve publication.</p>
<p id="status" class="counts"></p><small id="version"></small><p id="stale" class="notice" hidden>Older saved decisions belong to a different version and are excluded.</p>
</header><main><div class="toolbar"><input id="search" type="search" aria-label="Search claims and speakers" placeholder="Search claims or speakers"><select id="filter" aria-label="Filter review items"><option value="all">All flagged claims</option><option value="pending">Pending decisions</option><option value="contradicted">Contradicted</option><option value="unsupported">Unsupported</option><option value="uncertain">Uncertain</option></select><button id="download">Download decisions</button></div>
<p id="message" role="status"></p><details><summary>Audited wording corrections already applied</summary><div id="edits"></div></details><div id="cards"></div>
<p class="muted">Import the downloaded file with <code>crosstalk review-import &lt;run&gt; &lt;decisions.json&gt;</code>, then rebuild this screen to display recorded decisions.</p></main>
<script id="payload" type="application/json">__PAYLOAD__</script>
<script>
const data=JSON.parse(document.getElementById('payload').textContent), storageKey='crosstalk-review:'+JSON.stringify(data.lineage);
let savedDraft={};try{savedDraft=JSON.parse(localStorage.getItem(storageKey)||'{}')}catch(e){}
const decisions=new Map(data.decisions.map(d=>[d.id,d]));for(const [id,d] of Object.entries(savedDraft)){if(data.flags.some(f=>f.id===id))decisions.set(id,d)}
const el=(tag,text,cls)=>{const e=document.createElement(tag);if(text!==undefined)e.textContent=text;if(cls)e.className=cls;return e};
const persist=()=>{try{localStorage.setItem(storageKey,JSON.stringify(Object.fromEntries(decisions)))}catch(e){}updateStatus()};
const resolved=f=>{const d=decisions.get(f.id);return d&&d.note.trim()&&['accept_source','opinion'].includes(d.decision)&&(d.decision!=='accept_source'||d.passage_ids.length>0)};
function updateStatus(){document.getElementById('status').textContent=`${data.flags.filter(resolved).length} / ${data.flags.length} claims have a support or opinion decision · ${decisions.size} decisions drafted or recorded`;}
document.getElementById('topic').textContent=data.topic;
document.getElementById('version').textContent='Transcript version: '+data.lineage.transcript_hash;
document.getElementById('stale').hidden=!data.stale_decisions;
for(const edit of data.edits){const box=el('article');box.append(el('h2','Turn '+edit.idx),el('p',edit.reason,'muted'),el('p','Before: '+edit.before),el('p','After: '+edit.after));document.getElementById('edits').append(box)}
function render(){const target=document.getElementById('cards');target.replaceChildren();const query=document.getElementById('search').value.toLowerCase(),filter=document.getElementById('filter').value;
const flags=data.flags.filter(f=>(f.assertion+' '+f.speaker+' '+f.spoken_text).toLowerCase().includes(query)&&(filter==='all'||filter==='pending'&&!resolved(f)||filter===f.verdict));
for(const f of flags){const card=el('article');card.append(el('div',f.verdict+' · Turn '+f.idx+' · '+f.speaker,'tag'),el('h2',f.assertion),el('blockquote',f.spoken_text));if(f.editorial_note)card.append(el('p',f.editorial_note,'muted'));
const draft=decisions.get(f.id)||{id:f.id,decision:'',note:'',passage_ids:[],proposed_text:''};
const select=el('select',undefined,'decision');select.setAttribute('aria-label','Decision for '+f.assertion);for(const [value,title] of [['','Choose a decision'],['accept_source','Accept as source-supported, within stated scope'],['opinion','Policy opinion, question, or transcript summary'],['needs_correction','Needs a wording correction'],['defer','Defer for more evidence']]){const option=el('option',title);option.value=value;select.append(option)}select.value=draft.decision;card.append(select);
const rationale=el('textarea');rationale.value=draft.note;rationale.placeholder='Explain your decision and its scope';rationale.setAttribute('aria-label','Review rationale');card.append(el('label','Rationale (required)'),rationale);
const proposal=el('textarea');proposal.value=draft.proposed_text;proposal.placeholder='Proposed spoken wording, preserving this speaker’s position';proposal.setAttribute('aria-label','Proposed correction');card.append(el('label','Proposed correction (required for a correction decision)'),proposal);
const sources=el('details');sources.append(el('summary',f.passages.length+' retrieved source passages'));
for(const p of f.passages){const label=el('label',undefined,'passage'),check=el('input');check.type='checkbox';check.checked=draft.passage_ids.includes(p.id);check.setAttribute('aria-label','Cite '+p.id);label.append(check,el('span',' '+p.id+' · '+p.source.title));try{const url=new URL(p.source.url);if(['https:','http:'].includes(url.protocol)){const link=el('a','Open source');link.href=url.href;link.target='_blank';link.rel='noopener noreferrer';label.append(el('br'),link)}}catch(e){}label.append(el('p',p.text));check.onchange=()=>{draft.passage_ids=[...sources.querySelectorAll('input:checked')].map(i=>i.dataset.pid);commit()};check.dataset.pid=p.id;sources.append(label)}card.append(sources);
function commit(){draft.decision=select.value;draft.note=rationale.value;draft.proposed_text=proposal.value;if(draft.decision)decisions.set(f.id,draft);else decisions.delete(f.id);persist()}
select.onchange=commit;rationale.oninput=commit;proposal.oninput=commit;target.append(card)}if(!flags.length)target.append(el('p','No claims match this filter.','empty'));updateStatus()}
document.getElementById('search').oninput=render;document.getElementById('filter').onchange=render;
document.getElementById('download').onclick=()=>{const rows=[...decisions.values()];const invalid=rows.find(d=>!d.note.trim()||d.decision==='accept_source'&&!d.passage_ids.length||d.decision==='needs_correction'&&!d.proposed_text.trim());const message=document.getElementById('message');if(!rows.length||invalid){message.textContent='Add a rationale for every decision, source passages for accepted support, and proposed text for corrections.';return}const blob=new Blob([JSON.stringify({schema_version:1,lineage:data.lineage,decisions:rows},null,2)],{type:'application/json'}),url=URL.createObjectURL(blob),link=el('a');link.href=url;link.download='crosstalk-review-decisions.json';link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);message.textContent='Downloaded. Import this file to record these decisions in the project.'};render();
</script></html>"""


def build_screen(root: Path):
    data = snapshot(root)
    data["topic"] = json.loads((root / "dossier.json").read_text())["topic"]
    encoded = (
        json.dumps(data).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    )
    path = root / "review/index.html"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(PAGE.replace("__PAYLOAD__", encoded))
    return path
