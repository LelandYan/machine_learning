from initdata import db
import pickle

dbfile = open('people_pickle', 'wb')
pickle.dump(db, dbfile)
dbfile.close()
