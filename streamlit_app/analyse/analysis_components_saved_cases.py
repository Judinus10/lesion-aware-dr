def render_saved_case_card(case: dict, idx: int) -> str:
    patient_id = case.get("patient_id") or "-"
    patient_name = case.get("patient_name") or "-"
    age = case.get("age") or "-"
    saved_at = case.get("saved_at") or "-"
    gender = case.get("gender") or "-"
    mode = str(case.get("analysis_input_mode", "-")).title()
    summary = case.get("summary") or "-"

    return f"""
    <div class="saved-card">
        <div class="saved-grid">
            <div class="saved-col">
                <div class="saved-title">Case {idx + 1}</div>
                <div class="saved-row"><span class="saved-label">Patient ID:</span> {patient_id}</div>
                <div class="saved-row"><span class="saved-label">Patient Name:</span> {patient_name}</div>
                <div class="saved-row"><span class="saved-label">Age:</span> {age}</div>
            </div>
            <div class="saved-col">
                <div class="saved-row"><span class="saved-label">Saved:</span> {saved_at}</div>
                <div class="saved-row"><span class="saved-label">Gender:</span> {gender}</div>
                <div class="saved-row"><span class="saved-label">Mode:</span> {mode}</div>
                <div class="saved-summary"><span class="saved-label">Summary:</span> {summary}</div>
            </div>
        </div>
    </div>
    """