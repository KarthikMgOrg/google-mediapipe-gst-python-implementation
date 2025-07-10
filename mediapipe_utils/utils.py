
def group_gazes(data):
    result = []
    for k, v in data.items():
        float_k = round(float(k), 2)
        found_close = False
        for existing in result:
            if abs(float_k - existing) <= 1:
                found_close = True
                break

        if not found_close:
            result.append(float_k)
    return result