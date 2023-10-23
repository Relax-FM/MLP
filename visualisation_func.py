def average(labels, results):
    avg_lbl = sum(labels) / len(labels)
    avg_res = sum(results) / len(results)

    print(f"Average for label_tensor is : {avg_lbl}\nAverage for result_tensor is : {avg_res}")

    return avg_lbl, avg_res


def calculated_standard_deviation(labels, results):
    if len(labels) != len(results):
        raise ValueError('The lists must have the same length for calculating standard deviation')

    n = len(labels)
    squared_diff_sum = 0
    for i in range(n):
        squared_diff_sum += (results[i] - labels[i]) ** 2

    mean_squared_diff = squared_diff_sum/n
    standard_deviation = (mean_squared_diff)**0.5

    print(f'Standard deviation is : {standard_deviation}')
    return standard_deviation


def calculated_error(standard_deviation, avg_lbl):
    error = (standard_deviation/avg_lbl)*100
    print(f'Error is : {error}%')
    return error


def calculated_max_error(labels, results):

    if len(results) != len(labels):
        raise ValueError('The lists must have the same length for calculating standard deviation')

    max_error = (abs(results[0] - labels[0])/labels[0]) * 100
    for i in range(1, len(results)):
        now = (abs(results[i] - labels[i])/labels[i]) * 100
        if now > max_error:
            max_error = now

    print(f'Max error is : {max_error}%')
    return max_error
