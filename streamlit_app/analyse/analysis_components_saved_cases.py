def render_saved_case_card(case: dict, idx: int) -> str:
    return f"""
    <div class="saved-card">
        <div class="saved-grid">
            <div class="saved-col">
                <div><span class="saved-label">Case:</span> {idx + 1}</div>
                <div><span class="saved-label">Patient ID:</span> {case.get("patient_id", "-")}</div>
                <div><span class="saved-label">Patient Name:</span> {case.get("patient_name", "-")}</div>
                <div><span class="saved-label">Age:</span> {case.get("age", "-")}</div>
            </div>
            <div class="saved-col">
                <div><span class="saved-label">Saved:</span> {case.get("saved_at", "-")}</div>
                <div><span class="saved-label">Gender:</span> {case.get("gender", "-")}</div>
                <div><span class="saved-label">Mode:</span> {case.get("analysis_input_mode", "-").title()}</div>
                <div><span class="saved-label">Summary:</span> {case.get("summary", "-")}</div>
            </div>
        </div>
    </div>
    """