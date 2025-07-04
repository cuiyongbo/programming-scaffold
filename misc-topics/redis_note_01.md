# in redis, where do keys with ttl store, do redis store them with keys without ttl together?

Great question! Understanding how Redis handles keys with TTL (time-to-live) internally helps clarify its expiration mechanism and performance.

---

## Where does Redis store keys with TTL?

- **All keys (with or without TTL) are stored together in the main keyspace**, which is a dictionary (hash table) mapping keys to their values.
- Redis **does not separate keys with TTL into a different primary data structure**.
- Instead, TTL information is stored **separately** in a dedicated dictionary called the **expires dictionary** (`expires`), which maps keys to their expiration timestamps (in milliseconds).

---

## Summary of storage:

| Data Structure       | Contents                                    |
|----------------------|---------------------------------------------|
| **Main keyspace**    | All keys and their associated values        |
| **Expires dictionary**| Only keys that have TTL and their expire time |

---

## How does this work in practice?

- When you set a TTL on a key (e.g., with `EXPIRE` or `SET key value EX seconds`), Redis:
  - Stores the key and value in the main keyspace.
  - Adds an entry in the expires dictionary with the key and its expiration timestamp.
- Keys without TTL are **not** present in the expires dictionary.

---

## Why separate TTL info?

- This design keeps the main keyspace simple and efficient.
- TTL lookups and expiration checks are done by consulting the expires dictionary.
- Redis uses this to efficiently manage expirations without affecting normal key access.

---

## Expiration mechanism overview

- Redis uses **lazy expiration**: keys are checked for expiration only when accessed.
- Redis also performs **active expiration**: a background task samples keys with TTL and deletes expired ones proactively.
- Both mechanisms use the expires dictionary to find keys with TTL.

---

## Summary

- Keys with and without TTL **live together in the main keyspace**.
- TTL metadata is stored **separately** in the expires dictionary.
- This separation allows efficient expiration management without complicating normal key storage.

---

If you want, I can explain how Redis implements expiration algorithms or how the expires dictionary is structured internally!


# tell me  how the expires dictionary is structured internally in redis

Great question! Understanding how Redis internally structures the **expires dictionary** helps clarify how Redis efficiently manages key expirations.

---

## What is the expires dictionary in Redis?

- The **expires dictionary** is a separate hash table that stores **only keys with an associated TTL**.
- It maps keys to their **expiration timestamps** (in milliseconds since the Unix epoch).
- This dictionary is distinct from the main keyspace dictionary.

---

## Internal Structure

### 1. **Data Structure Type**

- The expires dictionary is implemented as a **hash table** (similar to the main keyspace).
- Keys in this dictionary are the **same Redis keys** (strings or binary-safe keys).
- Values are **64-bit integers** representing the absolute expiration time in milliseconds.

### 2. **Hash Table Details**

- Redis uses a **dynamic hash table** implementation with:
  - **Two hash tables** for incremental rehashing (to avoid blocking).
  - Open addressing with chaining for collision resolution.
- The expires dictionary follows the same implementation as the main dictionary (`dict.c`).

### 3. **Key-Value Pairs**

- **Key:** Redis key (pointer to the same key object used in main dictionary).
- **Value:** Expiration timestamp (`long long` integer, milliseconds since epoch).

---

## How it works together with the main keyspace

- When a key is set with TTL, Redis inserts/updates an entry in the expires dictionary.
- When the TTL is removed (e.g., `PERSIST` command), the key is removed from the expires dictionary.
- When a key is deleted or expires, Redis removes it from both the main dictionary and the expires dictionary.

---

## Why a separate dictionary?

- Keeps TTL management **decoupled** from key-value storage.
- Enables efficient lookup of expiration times without scanning the entire keyspace.
- Simplifies expiration checks and active expiration algorithms.

---

## Expiration Check Workflow

- When accessing a key, Redis checks if the key exists in the expires dictionary.
- If yes, Redis compares the current time with the stored expiration time.
- If expired, Redis deletes the key from both dictionaries and returns a miss.

---

## Summary Table

| Aspect               | Description                              |
|----------------------|------------------------------------------|
| Data structure       | Hash table (dict)                        |
| Keys                 | Redis keys with TTL                      |
| Values               | Expiration timestamps (ms since epoch)  |
| Relation to main dict | Separate from main keyspace dict         |
| Purpose              | Efficient TTL lookup and expiration      |

---

If you want, I can also explain how Redis performs **active expiration** using this dictionary or how incremental rehashing works!

# we already know redis's main keyspace is a hash table, which resolution policy does Redis use when collisions happen

Great question! Understanding how Redis handles hash collisions in its main keyspace is important for grasping its performance characteristics.

---

## Collision Resolution in Redis Hash Tables

Redis uses **separate chaining** (also called **open hashing**) as its collision resolution strategy.

### What does this mean?

- Each bucket in the hash table contains a **linked list** (chain) of entries that hash to the same bucket.
- When multiple keys collide (i.e., have the same hash modulo the table size), they are stored in this linked list.
- To find a key, Redis hashes the key to a bucket and then traverses the linked list in that bucket to find the matching key.

---

## Details

- Redis hash tables are implemented in `dict.c`.
- Each hash table is an array of pointers to **dictEntry** structs.
- Each **dictEntry** contains:
  - The key
  - The value
  - A pointer to the next entry in the chain (linked list)
- When inserting a new key:
  - Compute the bucket index.
  - Insert the new entry at the head of the linked list for that bucket.
- When searching:
  - Compute the bucket index.
  - Traverse the linked list to find the matching key.

---

## Why separate chaining?

- Simple to implement and maintain.
- Performance degrades gracefully with collisions.
- Supports dynamic resizing and incremental rehashing.

---

## Performance Considerations

- Ideally, the hash function distributes keys evenly to minimize chain length.
- Redis uses **MurmurHash2** (or SipHash for small keys) to generate good hash distributions.
- When the load factor (number of entries / number of buckets) grows too large, Redis **rehashes** to a bigger table to keep chains short.

---

## Summary

| Aspect                      | Description                         |
|-----------------------------|-----------------------------------|
| Collision resolution policy | Separate chaining (linked lists)  |
| Data structure per bucket   | Linked list of dictEntry structs  |
| Insertion                   | Insert at head of linked list      |
| Lookup                     | Traverse linked list in bucket     |
| Hash function               | MurmurHash2 or SipHash             |

---

If you want, I can explain how **incremental rehashing** works in Redis or how the hash functions are chosen!

# Redis also has a data structure called hash table, which resolution policy does it use then?

Great follow-up! In Redis, there are two different concepts of “hash tables” that sometimes cause confusion:

1. **Redis main keyspace dictionary** (the global key-to-value mapping).
2. **Redis Hash data type** (a Redis data structure that stores multiple field-value pairs under one Redis key).

---

## Clarifying the question

- The **main keyspace** is a dictionary mapping Redis keys (strings) to Redis objects (strings, lists, hashes, etc.).
- The **Redis Hash data type** is a data structure inside Redis that stores multiple field-value pairs, like a map/dictionary inside a key.

---

## Collision resolution in the Redis **main keyspace dictionary**

- Uses **separate chaining** with linked lists (as explained before).

---

## Collision resolution in the Redis **Hash data type**

Redis Hashes are implemented internally using **two possible encodings**:

### 1. **Ziplist encoding** (compact list)

- Used when the hash is small (few fields, small field/value sizes).
- It is a **compact, sequential list** — no hashing or collision resolution here.
- Lookup is linear scan — efficient for small hashes.

### 2. **Hashtable encoding**

- Used when the hash grows beyond thresholds.
- Implemented as a **hash table with separate chaining** (linked lists).
- Similar to the main keyspace dictionary, it uses **linked lists** to resolve collisions.

---

### Summary for Redis Hash data type:

| Encoding       | Collision Resolution Policy               |
|----------------|------------------------------------------|
| Ziplist        | No hashing; linear scan                   |
| Hashtable      | Separate chaining with linked lists      |

---

## Additional notes

- Redis automatically switches from ziplist to hashtable encoding when the hash grows large.
- Both the main dictionary and hash data type’s hashtable use the **same dict.c implementation** internally.
- The hash function used is the same (MurmurHash2 or SipHash).

---

## Summary Table

| Data Structure       | Collision Resolution Policy             |
|----------------------|----------------------------------------|
| Main keyspace dict   | Separate chaining (linked lists)       |
| Redis Hash (ziplist) | Linear scan (no hashing)                |
| Redis Hash (hashtable) | Separate chaining (linked lists)     |

---

If you want, I can provide more details on how Redis switches encoding or how the hash table is resized!

