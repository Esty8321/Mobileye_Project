git init# Data Yarok Research

## Selected Datasets

### PlantVillage (Original)
- **Source**: Kaggle – PlantVillage Dataset  
- **Content**: ~20,000 images of healthy and diseased leaves.  
- **Crops**: Tomato, Pepper, Potato, and more.  
- **Pros**: Very broad base, useful for classification.  
- **Cons**: Relatively old dataset, does not always include advanced annotations.  

### PlantDoc
- **Source**: PlantDoc – Paper & Dataset  
- **Content**: Diverse dataset with plants such as Cassava, Mango, Papaya.  
- **Pros**: Real-world ("in-the-wild") photos, wide variety of diseases.  
- **Cons**: Relatively small in size.  

### Plant Pathology 2020 (Apple Leaves)
- **Source**: Kaggle – Plant Pathology 2020  
- **Content**: ~3,600 images of apple leaves (Healthy, Apple scab, Cedar rust, Black rot).  
- **Pros**: High quality, well organized, includes "Healthy" category.  
- **Cons**: Focused only on apples.  

### PlantVillage (Extended)
- **Source**: Kaggle – PlantVillage Extended  
- **Content**: Extended version with 38 different categories.  
- **Size**: ~2.5GB.  
- **Pros**: Wide variety of crops and diseases, includes Healthy and Diseased.  
- **Cons**: Large dataset, requires more resources for processing.  

---

## Data Schema

| Field        | Type   | Description                          |
|--------------|--------|--------------------------------------|
| image_id     | String | Unique ID for each image             |
| file_path    | String | Path or URL of the image             |
| label        | String | Category: Healthy / Diseased         |
| crop_type    | String | Type of crop (tomato, potato, etc.)  |
| disease_type | String | Specific disease if applicable       |

---

## Database Choice

- **Images**: Store in object storage (e.g., AWS S3, Google Cloud Storage).  
- **Metadata**: Store in relational database (e.g., PostgreSQL) for queries and structure.  
- This combination provides scalability for large image data and flexibility for metadata management.
