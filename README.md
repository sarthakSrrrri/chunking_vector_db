

First create a database schema for because Milvus DB should know , what kind of feild(column_name) and datatype we want to store.



### Collection

A collection is the main storage unit in Milvus. It is where Milvus keeps a set of entities that follow the same structure.

You can compare it to a table in a traditional database, but a Milvus collection is designed to work with vector data as well.

For your RAG system, if you have 100 PDFs, you could create one collection called `pdf_chunks`. All the chunks generated from those PDFs can be stored in that collection.

A collection is useful because all entities inside it follow a defined schema. This means Milvus knows what kind of information each entity contains and which field contains the vector that should be searched.

A collection is therefore not a search algorithm. It is primarily the **logical container for your data**.

---

### Entity

An entity is one individual record stored inside a collection.

In a normal relational database, you would usually call this a row. In Milvus terminology, it is an entity.

For example, one entity could represent one chunk from a PDF. That entity could contain the chunk's ID, document ID, page number, text, and embedding vector.

The entity is the actual piece of data that Milvus stores and eventually returns when you perform a search.

The important distinction is that an entity is not necessarily just the vector. An entity can contain the vector along with other information that describes the vector.

---

### Field

A field is an individual attribute of an entity.

If an entity represents a PDF chunk, you might have fields such as `id`, `document_id`, `page`, `text`, and `embedding`.

Each field has a specific data type. For example, an ID might be an integer, the page number might be an integer, the text might be a string, and the embedding might be a vector.

Fields allow Milvus to understand what each piece of information represents.

There are also different kinds of fields in Milvus. Some fields contain normal scalar data, while a vector field contains vector data used for similarity search.

So when you hear "field," simply think **one property of your stored record**.

---

### Schema

A schema is the definition of the structure of a collection.

Before Milvus can properly store your entities, it needs to know what fields exist, what their names are, and what types of data they contain. The schema defines these rules.

For example, you might define that your collection has an integer ID, a string document ID, an integer page number, a string containing the chunk text, and a floating-point vector with a particular dimension.

The schema is therefore similar to the structure or contract of your collection.

It also helps maintain consistency. If your vector field is defined as having dimension 768, the vectors you insert into that field need to have 768 values.

So:

**Collection = where the data is kept.**

**Schema = what the data inside that collection is supposed to look like.**

---

### Primary Key

A primary key is the unique identifier of an entity.

Its purpose is identification, not similarity.

Suppose you have thousands of PDF chunks. You need some way to distinguish one chunk from another. The primary key provides that identity.

The primary key could be an automatically generated integer or a value you provide yourself, depending on how you design the collection.

It is important not to confuse the primary key with the vector.

The primary key answers:

> "Which entity is this?"

The vector answers a completely different question:

> "What numerical representation represents this content?"

The primary key doesn't determine whether two pieces of text are semantically similar.

---

### Vector

A vector is a list of numerical values that represents some data in a mathematical space.

In a RAG system, an embedding model takes your text and converts it into a vector.

For example, you might have a sentence about home-loan eligibility. The embedding model converts that sentence into hundreds or thousands of numerical values.

The purpose isn't for a human to understand those numbers individually. Their usefulness comes from how vectors relate to each other.

Texts with similar meanings tend to produce vectors that are relatively close according to the chosen similarity measure.

This allows Milvus to perform semantic retrieval rather than just looking for exact words.

For example, a query asking about "minimum age for a home loan" could retrieve a chunk saying "Applicants must be at least 21 years old," even though the wording isn't identical.

That is one of the main reasons vectors are useful for RAG.

---

### Vector Dimension

Vector dimension means the number of numerical values contained in a vector.

If a vector contains 768 numbers, its dimension is 768.

The dimension normally comes from the embedding model you use.

Different embedding models can produce vectors of different dimensions. Milvus needs to know the dimension of your vector field so it can correctly store and process the vectors.

Dimension does not mean the quality of the embedding.

A 1536-dimensional embedding isn't automatically better than a 768-dimensional embedding. The quality depends on the embedding model, its training, and how well it represents your particular data and queries.

For Milvus, dimension is primarily a **structural property of the vector**.

---

### Index

An index is a data structure used to make vector searching more efficient.

Without an appropriate search structure, finding the nearest vectors can require comparing the query against a very large number of stored vectors.

As your dataset grows, that can become computationally expensive.

A vector index organizes the stored vectors in a way that allows Milvus to find likely relevant vectors more efficiently.

There are different indexing approaches, such as HNSW and IVF. They use different mathematical and structural strategies for reducing the amount of work required during search.

The important concept for now is:

An index does not change your original vector or your text. It is an additional structure created to make searching those vectors faster.

---

### Metric

A metric is the mathematical method used to determine how close or similar two vectors are.

This is necessary because Milvus needs some definition of "similar."

For example, two vectors can be compared using cosine similarity, Euclidean distance (L2), or inner product (IP).

Different metrics measure relationships between vectors differently.

For semantic text retrieval, cosine similarity is commonly used because it focuses on the direction of the vectors rather than simply their absolute magnitude.

The metric should generally be chosen consistently with how your embedding model is intended to be used.

So the metric answers:

> "How should we mathematically compare these two vectors?"

---

### Search

Search is the operation of finding entities whose vectors are most relevant to a query vector.

The query normally starts as natural language.

For example:

"Who is eligible for this loan?"

Your embedding model converts that question into a vector.

Milvus then uses the vector field, index, and metric to find stored entities whose vectors are close or similar to the query vector.

The result isn't normally just a similarity score. You can retrieve the associated entity information as well, such as the original text, document ID, page number, and other metadata.

In a RAG system, those retrieved chunks become the context that is given to the LLM.

---

### Top-K

Top-K simply means the number of best search results you want.

If K is 5, you are asking Milvus to return the five best matching entities according to the search configuration.

Top-K is therefore a retrieval parameter, not a property of your collection.

Choosing K is an application-level decision. A small K gives you fewer candidate chunks, while a larger K gives you more information but can also introduce irrelevant content.

For RAG, you might retrieve several candidates and then use another component, such as a reranker, to select the most useful ones.

---

### One distinction to keep clear

There are three different ideas that are easy to mix up:

**Data:** your actual PDF chunks and their metadata.

**Vector:** the numerical representation of a chunk.

**Index:** the additional structure that helps Milvus search those vectors efficiently.

The vector is your searchable representation. The index is the mechanism that helps make searching that representation efficient.

That distinction will become important when we move to **HNSW, IVF, and ANN**.
