from datetime import datetime


info = {
    'w2v2': {
        'start': '2023-12-01 10:03:56',
        'end': '2023-12-02 02:18:38',
    },
    'bestrq-base': {
        'start': '2023-12-01 09:48:16',
        'end': '2023-12-01 16:31:51',
    },
    'w2v2-latter': {
        'start': '2023-12-05 16:15:20',
        'end': '2023-12-06 08:46:56',
    },
    'bestrq-latter': {
        'start': '2023-12-05 16:01:33',
        'end': '2023-12-05 22:58:16',
    },
    # 'bestrq-hc': {
    #     'start': '2023-12-01 09:54:39',
    #     'end': '2023-12-01 15:51:28',
    # },
    # 'bestrq-hb': {
    #     'start': '2023-12-01 11:43:22',
    #     'end': '2023-12-01 16:26:38',
    # },
}

start = datetime.strptime(info['w2v2']['start'], '%Y-%m-%d %H:%M:%S')
end = datetime.strptime(info['w2v2']['end'], '%Y-%m-%d %H:%M:%S')
w2v2_dif = end - start

dif_array = []
for key in info:
    print(key)
    start = datetime.strptime(info[key]['start'], '%Y-%m-%d %H:%M:%S')
    end = datetime.strptime(info[key]['end'], '%Y-%m-%d %H:%M:%S')
    dif = end - start
    dif_array.append(dif)
    print(dif)  # printed in default format
    # print((w2v2_dif - dif) / w2v2_dif)  # printed in default format
    # print(w2v2_dif / dif)  # printed in default format

wv_total = dif_array[0] + dif_array[2]
brq_total = dif_array[1] + dif_array[3]
print('wv_total')
print(wv_total)
print('brq_total')
print(brq_total)

print((wv_total - brq_total) / wv_total)
print(wv_total / brq_total)

print('wv_total GPU')
print(wv_total * 8)
print('brq_total GPU')
print(brq_total * 8)




