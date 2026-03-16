import dask
import time
from datetime import timedelta
from tqdm import tqdm
import numpy as np

def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))

if __name__ == '__main__':
    arr = np.arange(100000)

    datas = []
    start_time = time.time()
    for i in tqdm(range(10000)):
        ret = np.random.choice(arr, size=1000, replace=True)
        datas.append(ret)
    print('one process, replace=True, time usage {}'.format(get_time_dif(start_time)))

    datas = []
    start_time = time.time()
    for i in tqdm(range(10000)):
        ret = np.random.choice(arr, size=1000, replace=False)
        datas.append(ret)
    print('one process, replace=False, time usage {}'.format(get_time_dif(start_time)))
    
    datas = []
    start_time = time.time()
    for i in tqdm(range(10000)):
        ret = np.random.choice(arr[:10000], size=1000, replace=False)
        datas.append(ret)
    print('one process, 10000, replace=False, time usage {}'.format(get_time_dif(start_time)))


    start_time = time.time()
    lazy_results = []
    for i in tqdm(range(10000)):
        ret = dask.delayed(lambda x: np.random.choice(arr, size=1000, replace=False))(i)
        lazy_results.append(ret)
    datas = dask.compute(*lazy_results, scheduler='processes', num_workers=4)
    print(len(datas))
    print('dask 4 process, replace=False, time usage {}'.format(get_time_dif(start_time)))