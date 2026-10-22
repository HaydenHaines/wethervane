"use client";
import { useEffect, useId, useState } from "react";
import type { HouseGroupPublication } from "../lib/house-group-catalog";
import { HOUSE_GROUP_SAVED_VIEWS_STORAGE_KEY,MAX_HOUSE_GROUP_SAVED_VIEWS,createHouseGroupSavedView,houseGroupSavedViewAvailability,parseHouseGroupSavedViews,serializeHouseGroupSavedViews,verifyHouseGroupSavedView,type HouseGroupSavedView } from "../lib/house-group-saved-views";
export default function HouseGroupSavedViews({publication,raceId,mapView,comparison,busy,error,onRestore}:{publication:HouseGroupPublication;raceId:string;mapView:"forecast"|"actual";comparison:HouseGroupPublication|null;busy:boolean;error:string;onRestore:(view:HouseGroupSavedView)=>void}){
  const inputId=useId(),[views,setViews]=useState<HouseGroupSavedView[]>([]),[label,setLabel]=useState(""),[notice,setNotice]=useState(""),[storageReady,setStorageReady]=useState(false),[saving,setSaving]=useState(false),[exactStatuses,setExactStatuses]=useState<Record<string,{status:string;reason?:string}>>({});
  useEffect(()=>{
    function read(){try{setViews(parseHouseGroupSavedViews(localStorage.getItem(HOUSE_GROUP_SAVED_VIEWS_STORAGE_KEY)));setStorageReady(true);}catch{setStorageReady(false);setNotice("Saved group views cannot be read in this browser.");}}
    read();const changed=(event:StorageEvent)=>{if(event.key===HOUSE_GROUP_SAVED_VIEWS_STORAGE_KEY||event.key===null)read();};window.addEventListener("storage",changed);return()=>window.removeEventListener("storage",changed);
  },[]);
  useEffect(()=>{
    let current=true;setExactStatuses({});
    Promise.all(views.filter(v=>v.runId===publication.result.id&&(!v.comparison||v.comparison.runId===comparison?.result.id)).map(async view=>[view.id,await verifyHouseGroupSavedView(view,publication,comparison)] as const)).then(entries=>{if(current)setExactStatuses(Object.fromEntries(entries));});
    return()=>{current=false;};
  },[views,publication,comparison]);
  function persist(next:HouseGroupSavedView[],success:string){try{localStorage.setItem(HOUSE_GROUP_SAVED_VIEWS_STORAGE_KEY,serializeHouseGroupSavedViews(next));setViews(next);setNotice(success);return true;}catch{setNotice("This change could not be saved. Your browser's saved list has not changed.");return false;}}
  const reason=busy||saving?"Wait for the requested result to finish loading or saving.":error?"Resolve the requested result before saving.":!storageReady?"Browser storage is unavailable.":!publication.result.races.some(r=>r.id===raceId)?"Select a district before saving.":views.length>=MAX_HOUSE_GROUP_SAVED_VIEWS?"Delete a saved group view before adding another.":"";
  async function save(){if(reason||!label.trim())return;setSaving(true);try{const view=await createHouseGroupSavedView(label,publication,{raceId,mapView,comparison});if(persist([...views,view],`Saved ${view.label} in this browser.`))setLabel("");}catch{setNotice("The exact displayed group result or comparison could not be saved.");}finally{setSaving(false);}}
  return <details className="composition-saved house-saved-views">
    <summary>Saved group views <span aria-hidden="true">({views.length})</span></summary>
    <p className="small">Saved in this browser on this device. Experimental group views use a separate saved list.</p>
    <div className="save-view-form"><label htmlFor={inputId}>Name this group view</label><div><input id={inputId} maxLength={80} value={label} onChange={event=>setLabel(event.target.value)} onKeyDown={event=>{if(event.key==="Enter"){event.preventDefault();void save();}}}/><button type="button" disabled={!!reason||!label.trim()} onClick={()=>void save()}>Save group view</button></div></div>
    {reason&&<p className="small">{reason}</p>}{notice&&<p className="small" role="status">{notice}</p>}
    {views.length?<ul className="saved-view-list">{views.map(view=>{
      const preflight=houseGroupSavedViewAvailability(view,publication.catalog),exact=exactStatuses[view.id];
      const unavailable=preflight.status==="unavailable"||exact?.status==="unavailable";
      return <li key={view.id} className={unavailable?"house-saved-unavailable":""}><div className="house-saved-description"><button type="button" className="saved-view-restore" disabled={busy||saving||unavailable} onClick={()=>onRestore(structuredClone(view))}>{view.label}</button><p className="small">{view.cycle} · {view.runLabel} · {view.raceId.split(":")[1]} · {view.mapView==="actual"?"Actual winners":"Forecast"}</p>{view.comparison&&<p className="small">Compared with {view.comparison.runLabel}</p>}{unavailable?<p className="small"><strong>Unavailable:</strong> {preflight.reason||exact?.reason}</p>:<p className="small">The exact saved results are checked before restoration.</p>}</div><button type="button" className="saved-view-delete" aria-label={`Delete ${view.label}`} onClick={()=>persist(views.filter(v=>v.id!==view.id),`Deleted ${view.label}.`)}>Delete</button></li>;
    })}</ul>:<p className="small">{storageReady?"No group views saved in this browser yet.":"Saved group views are unavailable until browser storage can be read."}</p>}
  </details>;
}
