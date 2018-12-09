import shelve
from person_alternative import Person, Manager

fieldnames = ('name', 'age', 'job', 'pay')
db = shelve.open('class_shelve')
while True:
    key = input('\nKey ? => ')
    if not key: break
    if key in db:
        record = db[key]
    else:
        record = Person(name='?', age='?')
    for field in fieldnames:
        currval = getattr(record, field)
        newtext = input(f'\t[{field}]={currval}\n\t\tnew?=>')
        if newtext:
            setattr(record, field, eval(newtext))
    db[key] = record
db.close()
