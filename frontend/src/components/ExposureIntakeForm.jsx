import { createContext, useContext, useState } from "react";
import en from "../i18n/exposure_form_en.json";
import hi from "../i18n/exposure_form_hi.json";
import { S } from "../strings.js";

const TABLES = { en, hi };
const INIT = { age: "", fpu: "", fpy: "", scar: "", scarAge: "", ulcer: "", water: "", state: "", district: "", yws: "", hours: "", owy: "", sun: "", fam: "", tob: "", consent: false };

const num = (x) => (x === "" || x === null ? null : Number(x));
const intIn = (x, lo, hi) => { const n = num(x); return n !== null && Number.isInteger(n) && n >= lo && n <= hi; };

/** Builds the payload exactly matching ExposureHistory (exposure_schema.py). */
export function buildPayload(v) {
  const scar = v.scar === "yes";
  const hours = Number(v.hours);
  return {
    age_years: Number(v.age),
    fire_pot_use: v.fpu,
    fire_pot_years: v.fpu === "never" ? 0 : Number(v.fpy),
    has_burn_scar: scar,
    burn_scar_age_years: scar ? Number(v.scarAge) : null,
    burn_scar_nonhealing_ulcer: scar ? v.ulcer === "yes" : false,
    water_source: v.water,
    district: v.district.trim(),
    state: v.state.trim() || null,
    years_at_water_source: num(v.yws),
    outdoor_hours_per_day: hours,
    outdoor_work_years: hours > 0 ? Number(v.owy) : 0,
    sun_protection: v.sun,
    family_history_skin_cancer: v.fam === "yes",
    tobacco_use: v.tob,
  };
}

// Returns an i18n validation key or null.
function check(step, v) {
  const age = num(v.age);
  const req = "validation.required", rng = "validation.out_of_range", ex = "validation.exceeds_age";
  const dur = (x, lo, hi) => (!intIn(x, lo, hi) ? rng : Number(x) > age ? ex : null);
  switch (step) {
    case 0: return v.age === "" ? req : intIn(v.age, 0, 120) ? null : rng;
    case 1: return !v.fpu ? req : v.fpu === "never" ? null : dur(v.fpy, 1, 100);
    case 2: return !v.scar ? req : v.scar === "no" ? null : dur(v.scarAge, 0, 100) || (!v.ulcer ? req : null);
    case 3: return !v.water || !v.district.trim() ? req : v.yws === "" ? null : dur(v.yws, 0, 120);
    case 4: {
      const h = num(v.hours);
      if (h === null) return req;
      if (!(h >= 0 && h <= 24)) return rng;
      return (h > 0 && dur(v.owy, 0, 80)) || (!v.sun ? req : null);
    }
    default: return !v.fam || !v.tob ? req : null;
  }
}

const Ctx = createContext(null);

function Choice({ k, opts }) {
  const { v, set } = useContext(Ctx);
  return (
    <div className="choices" role="radiogroup">
      {opts.map(([val, label]) => (
        <label className="choice" key={val}>
          <input type="radio" name={k} value={val} checked={v[k] === val} onChange={() => set(k)(val)} />
          <span>{label}</span>
        </label>
      ))}
    </div>
  );
}
function Field({ id, i, children, type = "radio" }) {
  const { T } = useContext(Ctx);
  return (
    <div className="field" style={{ "--i": i }}>
      {type === "radio" ? <div className="lbl">{T[`${id}.label`]}</div> : <label htmlFor={id}>{T[`${id}.label`]}</label>}
      <div className="help">{T[`${id}.help`]}</div>
      {children}
    </div>
  );
}
function Num({ id, k, i, step = 1 }) {
  const { v, set } = useContext(Ctx);
  return (
    <Field id={id} i={i} type="num">
      <input className="input" id={id} type="number" inputMode="decimal" min="0" step={step} value={v[k]} onChange={(e) => set(k)(e.target.value)} />
    </Field>
  );
}

export default function ExposureIntakeForm({ lang, initial, onSubmit, onBack, error }) {
  const T = TABLES[lang], ui = S[lang];
  const [v, setV] = useState(initial || INIT);
  const [step, setStep] = useState(0);
  const [errKey, setErrKey] = useState(null);
  const set = (k) => (val) => { setV((p) => ({ ...p, [k]: val })); setErrKey(null); };
  const last = ui.steps.length - 1;

  const yn = [["yes", T["common.yes"]], ["no", T["common.no"]]];
  const opts = (prefix, keys) => keys.map((k) => [k, T[`${prefix}.option.${k}`]]);
  function next() {
    const e = check(step, v);
    if (e) return setErrKey(e);
    if (step < last) setStep(step + 1);
    else onSubmit(buildPayload(v), v.consent, v);
  }
  function back() { setErrKey(null); step === 0 ? onBack() : setStep(step - 1); }

  return (
    <Ctx.Provider value={{ v, set, T }}>
    <section className="card enter">
      <div className="progress" role="progressbar" aria-valuemin={1} aria-valuemax={ui.steps.length} aria-valuenow={step + 1}>
        <i style={{ width: `${((step + 1) / ui.steps.length) * 100}%` }} />
      </div>
      {step === 0 && <p>{T["form.intro"]}</p>}
      <h2>{ui.steps[step]}</h2>
      <div key={step} className="page">
        {step === 0 && <Num id="age_years" k="age" i={0} />}
        {step === 1 && (<>
          <Field id="fire_pot_use" k="fpu" i={0}><Choice k="fpu" opts={opts("fire_pot_use", ["never", "past", "current"])} /></Field>
          {v.fpu && v.fpu !== "never" && <Num id="fire_pot_years" k="fpy" i={1} />}
        </>)}
        {step === 2 && (<>
          <Field id="has_burn_scar" k="scar" i={0}><Choice k="scar" opts={yn} /></Field>
          {v.scar === "yes" && <>
            <Num id="burn_scar_age_years" k="scarAge" i={1} />
            <Field id="burn_scar_nonhealing_ulcer" k="ulcer" i={2}><Choice k="ulcer" opts={yn} /></Field>
          </>}
        </>)}
        {step === 3 && (<>
          <Field id="water_source" k="water" i={0}><Choice k="water" opts={opts("water_source", ["handpump_tubewell", "deep_borewell", "dug_well", "piped_treated", "bottled_or_filtered", "surface_water", "other_unknown"])} /></Field>
          <Field id="state" k="state" i={1} type="text"><input className="input" id="state" maxLength={100} value={v.state} onChange={(e) => set("state")(e.target.value)} /></Field>
          <Field id="district" k="district" i={2} type="text"><input className="input" id="district" maxLength={100} value={v.district} onChange={(e) => set("district")(e.target.value)} /></Field>
          <Num id="years_at_water_source" k="yws" i={3} />
        </>)}
        {step === 4 && (<>
          <Num id="outdoor_hours_per_day" k="hours" i={0} step={0.5} />
          {Number(v.hours) > 0 && <Num id="outdoor_work_years" k="owy" i={1} />}
          <Field id="sun_protection" k="sun" i={2}><Choice k="sun" opts={opts("sun_protection", ["regular", "sometimes", "never"])} /></Field>
        </>)}
        {step === 5 && (<>
          <Field id="family_history_skin_cancer" k="fam" i={0}><Choice k="fam" opts={yn} /></Field>
          <Field id="tobacco_use" k="tob" i={1}><Choice k="tob" opts={opts("tobacco_use", ["none", "smokeless", "smoking", "both"])} /></Field>
          <label className="consent field" style={{ "--i": 2 }}>
            <input type="checkbox" checked={v.consent} onChange={(e) => set("consent")(e.target.checked)} />
            <span>{ui.consent}</span>
          </label>
        </>)}
      </div>
      {(errKey || error) && <div className="err" role="alert">{errKey ? T[errKey] : error}</div>}
      <div className="btn-row">
        <button className="btn" onClick={back}>{ui.back}</button>
        <button className="btn btn-primary" onClick={next}>{step === last ? T["form.submit"] : ui.next}</button>
      </div>
    </section>
    </Ctx.Provider>
  );
}
