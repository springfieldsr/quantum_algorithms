import numpy as np
import  cirq
from Grover import Grover

def grover_random_test(number_of_qubits,  number_of_tests):

    def function(input):
        return 1 if input == answer else 0

    grover_result = []
    ground_truth = []
    for _ in range(number_of_tests):
        answer = np.random.randint(low=0,high=2**number_of_qubits)                                  # Randomly choose a key
        ground_truth.append(answer)
        grover = Grover(number_of_qubits, function)                                                 # Instantiate a Grover class
        result = grover.grover_solver()                                                             # Output result is string
        grover_result.append(int(result,2))

    cnt = 0
    try:
        assert grover_result == ground_truth                                                        # Hopefully, we can pass every test
    except:
        for i in range(number_of_tests):                                                            # Count how many times we failed
            if grover_result[i] != ground_truth[i]:
                cnt += 1
        print("In {} tests, our algorithm failed {} times.".format(number_of_tests, cnt))

        return cnt/number_of_tests
    print("Our Grover simulator passed {} tests.".format(number_of_tests))
    return cnt

print(grover_random_test(3,2))