% now we want to add exceptions (proleg like)
% holds(+Atom) : atom is derivable from facts and rules with exceptions
% holds(+Atom, -Trace) : same but with an explanantion trace (tree)
% applicable(+RuleId, -WhyNot) : rule RuleId can be applied
% supporting_rule(+Atom, +M, -RuleId) : atom has a supporting rule wrt interpretation M

% so we want a new rule: rule(C, H, Body) that is applicable iiff
% every item in body holds and no exception(C,E) such that E holds

% its broken because 
% 1 ?- holds(permitted_to_disclose(alice,bob)).
% returns false.
% rest seems to be fine? Also havent defined the holds/2
% coudlnt finish exo 2...

rule(r1, permitted_to_disclose(Doctor,Patient), 
     [doctor(Doctor), patient(Patient), treats(Doctor,Patient)]).

rule(e1, forbidden_to_disclose(Doctor,Patient), 
     [doctor(Doctor), patient(Patient), treats(Doctor,Patient), suspended(Doctor)]).

holds(Atom) :- 
    fact(Atom).

holds(Atom) :-
    rule(RuleId, Atom, Body),
    applicable(RuleId),
    holds_body(Body).

holds_body([B | T]) :-
    holds(B),
    holds_body(T).

holds_body([]). % true for empty body to end the recursion

applicable(RuleId) :- % applicable(+rudeid, -whynot) where whynot will be the returned info (string?) 
    \+ (exception(RuleId, Exception), holds(Exception)).

fact(doctor(alice)).
fact(doctor(charlie)).
fact(patient(bob)).
fact(treats(alice,bob)).
fact(suspended(charlie)). % charlie is suspended
fact(treats(charlie,bob)). % but still treats bob
exception(r1, forbidden_to_disclose(charlie,bob)). % idk if this is correct cuz i think i only want facts?