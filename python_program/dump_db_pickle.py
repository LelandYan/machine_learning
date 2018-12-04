import pickle
dbfile = open('people_pickle','rb')
db = pickle.load(dbfile)
for key in db:
    print(key,'=>',db[key])

