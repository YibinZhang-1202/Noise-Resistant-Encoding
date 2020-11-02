from nre_clustering import *


########################################################################################
DATA_PATH = 'dataset/dirty/dirty_itunes_amazon_exp_data'

META_DATA = 'encoded_cluster_0.4-1.pickle'
GEN_PATH = ''
MISSPELLED_TESTSET_PATH = ''

test_sets = ['valid.csv']
test_set_list = [pd.read_csv(os.path.join(DATA_PATH, test_sets[0]))]
# test_sets = os.listdir(os.path.join(DATA_PATH,MISSPELLED_TESTSET_PATH))
# test_set_list = [pd.read_csv(os.path.join(DATA_PATH, MISSPELLED_TESTSET_PATH + x)) for x in test_sets]

all_right_fields = [x for x in list(test_set_list[0])[2:] if x.startswith('right')]

with open(os.path.join(DATA_PATH, META_DATA), 'rb') as f:
    meta_data = pickle.load(f)

fields = meta_data['fields']


for x in fields:
    cluster_graph = meta_data[x]
    print(x)

    for i in range(0, len(test_set_list)):
        test_set_list[i] = encode_parallel(test_set_list[i], cluster_graph, x, all_right_fields)
        print("Encoding", test_sets[i], "done.")
    print()


for i in range(0, len(test_set_list)):
    test_set_list[i].to_csv(os.path.join(DATA_PATH + GEN_PATH, 'encoded_' + test_sets[i]), index=False)
