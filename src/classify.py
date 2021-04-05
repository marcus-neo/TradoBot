def Classify(close,number):
    output = [0] * len(close)
    for i in range(len(close)):
        if i >= number:
            basePrice = close[i]
            offset = 1
            position = 0
            while i + offset < len(close) - 1 and position == 0 and offset <= 5:
                offset += 1
                if close[i + offset] < 0.95 * close[i]:
                    position = -1
                elif close[i + offset] > 1.05 * close[i]:
                    position = 1
            output[i] = position
    return output