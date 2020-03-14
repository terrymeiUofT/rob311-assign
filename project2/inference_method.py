from collections import deque
from support import definite_clause

### THIS IS THE TEMPLATE FILE
### WARNING: DO NOT CHANGE THE NAME OF FILE OR THE FUNCTION SIGNATURE


def pl_fc_entails(symbols_list : list, KB_clauses : list, known_symbols : list, query : int) -> bool:
    """
    pl_fc_entails function executes the Propositional Logic forward chaining algorithm (AIMA pg 258).
    It verifies whether the Knowledge Base (KB) entails the query
        Inputs
        ---------
            symbols_list  - a list of symbol(s) (have to be integers) used for this inference problem
            KB_clauses    - a list of definite_clause(s) composed using the numbers present in symbols_list
            known_symbols - a list of symbol(s) from the symbols_list that are known to be true in the KB (facts)
            query         - a single symbol that needs to be inferred

            Note: Definitely check out the test below. It will clarify a lot of your questions.

        Outputs
        ---------
        return - boolean value indicating whether KB entails the query
    """

    ### START: Your code
    inferred = {}
    for symb in symbols_list:
        inferred.update({str(symb): False})
    count = {}
    for symb in known_symbols:
        count.update({str(symb): 0})
    for kb_clause in KB_clauses:
        symb = kb_clause.conclusion
        c = len(kb_clause.body)
        count.update({str(symb): c})

    while known_symbols:
        p = known_symbols.pop(0)
        if p == query:
            return True
        if inferred[str(p)] == False:
            inferred[str(p)] = True
            for c in KB_clauses:
                if p in c.body:
                    count[str(c.conclusion)] -= 1
                    if count[str(c.conclusion)] == 0:
                        known_symbols.append(c.conclusion)

    return False # remove line if needed
    ### END: Your code


# SAMPLE TEST
if __name__ == '__main__':

    # Symbols used in this inference problem (Has to be Integers)
    symbols = [1, 2, 9, 4, 5]

    # Clause a: 1 and 2 => 9
    # Clause b: 9 and 4 => 5
    # Clause c: 1 => 4
    KB = [definite_clause([1, 2], 9), definite_clause([9, 4], 5), definite_clause([1], 4)]

    # Known Symbols 1, 2
    known_symbols = [1, 2]

    # Does KB entail 5?
    entails = pl_fc_entails(symbols, KB, known_symbols, 5)

    print("Sample Test: " + ("Passed" if entails == True else "Failed"))
