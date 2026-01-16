% --- facts (scenario) ---
fact(doctor(alice)).
fact(doctor(charlie)).
fact(patient(bob)).
fact(treats(alice,bob)).
fact(info_medical(bob)).
fact(request_disclose(alice,bob)). % this means alice requests to disclose bob's medical info

% --- rules (no exceptions so far)
permitted_to_disclose(Doctor,Patient) :- 
    fact(doctor(Doctor)),
    fact(patient(Patient)),
    fact(treats(Doctor,Patient)),
    fact(info_medical(Patient)).
    % should we add request disclose between doctor and patient?

forbidden_to_disclose(Doctor,Patient) :- 
% we want forbid if doctor doesnt treat patient and thats it? (plus doctor and patient exist)
    fact(doctor(Doctor)),
    fact(patient(Patient)),
    \+ fact(treats(Doctor,Patient)).