%doctor can disclose if they treat the patient
rule(r1, permitted_to_disclose(Doctor,Patient),
     [doctor(Doctor), patient(Patient), treats(Doctor,Patient)]).

% doctor is forbidden to disclose if suspended
rule(e1, forbidden_to_disclose(Doctor,Patient),
     [doctor(Doctor), patient(Patient), treats(Doctor,Patient), suspended(Doctor)]).

% this is our default prohibition 
rule(c1, forbidden_to_disclose(alice,bob), [doctor(alice), patient(bob), treats(alice,bob), info_medical(bob)]).

% add new exception -> disclosure is allowed if patient consents
exception(c1, forbidden_to_disclose(alice,bob), consent(bob)).

% exception to exception -> consent is invalid if coerced
rule(c2, coerced_consent(bob), [coercion(bob)]).
exception(consent(bob), consent(bob), coerced_consent(bob)).

fact(doctor(alice)).
fact(doctor(charlie)).
fact(patient(bob)).
fact(treats(alice,bob)).
fact(treats(charlie,bob)).
fact(suspended(charlie)). % should trigger charlie not allowed to disclose bob execption
fact(info_medical(bob)).
fact(request_disclose(alice,bob)).

% variant A bob consents
fact(consent(bob)). % only this for variant A (and remove below)

% variant B bob is coerced to consent
% fact(coercion(bob)). % switch this with above for B

% if forbid disclose holds, then permit disclose is blocked
exception(r1, permitted_to_disclose(Doctor,Patient), forbidden_to_disclose(Doctor,Patient)).


% holds/1 just checks if atom is true (i.e a fact)
holds(A) :-
    fact(A).

% checks if atom is true through a rule (which should trigger holds/1 eventually)
holds(A) :-
    rule(RuleId, A, Body),
    body_holds(Body, _),
    \+ is_rule_defeated(RuleId, A).

% the recursion to check if all the body contents hold (or not)
body_holds([], []). % empty body holds to end recusion
body_holds([H|T], [HTrace|TTraces]) :-
    holds(H, HTrace),
    body_holds(T, TTraces).

% check if a rule is defeated by an exception
is_rule_defeated(RuleId, HeadAtom) :-
    exception(RuleId, HeadPattern, BlockingCondition),
    HeadPattern = HeadAtom, % check the rule name matches the one found in the exception
    % check condition of exception holds to trigger it
    holds(BlockingCondition).

% now we deal with traces
holds(A, fact(A)) :-
    fact(A).

% success trace which succeeds only if body holds AND rule is not defeated
holds(A, rule(RuleId, A, BodyTraces)) :-
    rule(RuleId, A, Body),
    body_holds(Body, BodyTraces),
    \+ is_rule_defeated(RuleId, A).

% defeated trace which shows that a rule was defeated by a blocking condition
% kinda same as before
holds(A, defeated(RuleId, by(BlockingCondition, BlockingTrace))) :-
    rule(RuleId, A, Body),
    body_holds(Body, _),  % og rule body succeeds
    exception(RuleId, HeadPattern, BlockingCondition),
    HeadPattern = A, 
    holds(BlockingCondition, BlockingTrace).  


% rule is applicable
applicable(RuleId, HeadAtom, applicable) :-
    rule(RuleId, HeadAtom, Body),
    body_holds(Body, _),
    \+ is_rule_defeated(RuleId, HeadAtom).

applicable(RuleId, _HeadAtom, missing(MissingAtom)) :-
    rule(RuleId, _, Body),
    member(MissingAtom, Body),
    \+ holds(MissingAtom).

applicable(RuleId, HeadAtom, blocked_by_exception(BlockingCondition)) :-
    rule(RuleId, HeadAtom, Body),
    body_holds(Body, _),  % original rule body holds
    exception(RuleId, HeadPattern, BlockingCondition),
    HeadPattern = HeadAtom,  % heads match
    holds(BlockingCondition).  % and block verified

supporting_rule(Atom, RuleId, Trace) :-
    holds(Atom, rule(RuleId, Atom, Trace)).