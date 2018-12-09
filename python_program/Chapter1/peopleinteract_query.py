import shelve

fieldname = ('name', 'age', 'job', 'pay')
maxfield = max(len(f) for f in fieldname)
db = shelve.open('class_shelve')

while True:
    key = input('\nKey ? => ')
    if not key: break
    try:
        record = db[key]
    except:
        print(f'No such key {key}!')
    else:
        for field in fieldname:
            print(field.ljust(maxfield), "=>", getattr(record, field))
