import time
import os

# sqlite-based cache
class cache_sqlite:

    connection = None
    cursor = None
    tables = []

    def __init__(self):

        
        self.connection = sqlite3.connect(':memory:', check_same_thread=False, isolation_level=None)
        self.cursor = self.connection.cursor()
        self.cursor.execute("PRAGMA journal_mode=wal")
        self.cursor.execute("PRAGMA wal_checkpoint=TRUNCATE")

        
        self.clean_cache()


    
    def clean_cache(self):

        
        Timer(30, self.clean_cache).start()

        
        for table in self.tables:
            self.cursor.execute("DELETE FROM {} WHERE expires < ?".format(table), (int(time.time()),))

    
    def get(self, pool, key):

        
        if pool not in self.tables:
            return None

        
        result = self.cursor.execute("SELECT value FROM {} WHERE key = ?".format(pool), (key,)).fetchone()

        
        if not result:
            return None

        
        return pickle.loads(result[0])

    
    def set(self, pool, key, value, ttl=0):

        
        if pool not in self.tables:
            self.cursor.execute("CREATE TABLE IF NOT EXISTS {}(key TEXT PRIMARY KEY, value TEXT, expires INTEGER)".format(pool))
            self.cursor.execute("CREATE INDEX expires_{0} ON {0} (expires ASC)".format(pool))
            self.tables.append(pool)

        
        self.cursor.execute("REPLACE INTO {} VALUES (?, ?, ?)".format(pool), (key, pickle.dumps(value), int(time.time() + ttl) if ttl > 0 and ttl <= 2592000 else ttl))


class cache_memcached:

    client = None
    prefix = 'sentiment'

    def __init__(self):
        
        self.client = memcache.Client(['localhost:11211'])

    # get cache element
    def get(self, pool, key):
        
        return self.client.get(self.prefix + '##' + pool + '##' + key.encode('ascii', 'xmlcharrefreplace').decode('ascii'))

    # set element in cache
    def set(self, pool, key, value, ttl=0):
        self.client.set(self.prefix + '##' + pool + '##' + key.encode('ascii', 'xmlcharrefreplace').decode('ascii'), value, ttl)


# import variable
cache = None


if os.environ.get('dev', False):
    import sqlite3
    from threading import Timer
    import pickle

    cache = cache_sqlite()
# else - memcached
else:
    import memcache

    cache = cache_memcached()
