## Notes about Spacy's Relation Extraction between Entities
Source: [YouTube](https://www.youtube.com/watch?v=8HL-Ap5_Axo&t=358s)

REL = `RELation Extraction` = Semantic Relationship between Entities
![image](https://user-images.githubusercontent.com/24909551/161043268-a043f545-f640-420a-9af0-60514580507b.png)

Bio-Medical Example: 
![image](https://user-images.githubusercontent.com/24909551/161043458-e5acda14-0027-4575-b91e-306208328810.png)

How REL will be added to Spacy's Pipeline: 
![image](https://user-images.githubusercontent.com/24909551/161043606-5c2fdc5b-04fd-4745-98b5-b412283aebfe.png)

How 2 entities are related is represented?
If entity 1 is related to entity 2: `[token vector of entity 1, token vector of entity 2]`
![image](https://user-images.githubusercontent.com/24909551/161044011-68df0f27-6c23-4911-be5d-b8e954785c85.png)

- Next steps: (3 types of REL classes - `BINDING`, `ACTIVATION`, `INHIBITION`)
- Above 0.5, is considered True
![image](https://user-images.githubusercontent.com/24909551/161044448-38fae140-dc86-47f0-bba1-8fa92c93ac7e.png)

**REL Pipeline**:

![image](https://user-images.githubusercontent.com/24909551/161044816-da43e479-a4b4-4991-8f8e-220f33e257eb.png)

A More example (multi-word tokens):

![image](https://user-images.githubusercontent.com/24909551/161047460-5deb131a-acd3-4794-8993-c177a18cdd4d.png)
![image](https://user-images.githubusercontent.com/24909551/161047566-65a9b6b7-9343-43d9-aa0f-299450d4bc16.png)

A Typical Config line: 
![image](https://user-images.githubusercontent.com/24909551/161047778-77c0253a-9f43-4500-ba73-f286afc11dcf.png)

Snippets from Config.cfg
![image](https://user-images.githubusercontent.com/24909551/161047897-2d0362f4-da81-4bb2-8459-b8ac8ae479b8.png)
- the model used in tok2vec (embedding creation) is `HashEmbedCNN`
![image](https://user-images.githubusercontent.com/24909551/161048056-151ba4aa-8dcd-4f04-9132-e4e2ca5fdca8.png)
![image](https://user-images.githubusercontent.com/24909551/161048462-58f3b045-b559-4766-a4b7-1c28a107fd32.png)

How to use/train the REL in spacy: 
```python
doc._.rel
```
![image](https://user-images.githubusercontent.com/24909551/161049370-c02da1a6-2128-4922-9bc5-3238fabeb7d9.png)

**What happens when we call `nlp(text)`**:
![image](https://user-images.githubusercontent.com/24909551/161049571-c084a4dd-59ee-4ebd-9480-fc159f085d57.png)

**Replicate REL extraction using `spacy project`**
```bash
spacy project clone tutorials/rel_component
```
![image](https://user-images.githubusercontent.com/24909551/161049985-0a826a6b-40d8-4e1f-99a2-0c74ccff987b.png)




