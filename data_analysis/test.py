# _*_ coding: utf-8 _*_

def invert_dictionary(dict_sample):
    int_list = []
    for i in dict_sample.values():
        if isinstance(i,list):
            int_list.extend(i)
        else:
            int_list.append(i)
    int_list = set(int_list)
    dict_result = {i:[] for i in int_list}
    for i in int_list:
        for key,value in dict_sample.items():
            if i in value:
                dict_result.get(i,[]).append(key)
    return dict_result
if __name__ == '__main__':
    dict_sample = {'a': [1, 2, 3], 'b': [1, 2], 'z': [5]}
    res = invert_dictionary(dict_sample)
    print(res)