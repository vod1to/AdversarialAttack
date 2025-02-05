from deepface import DeepFace

models = [
  "VGG-Face", 
  "Facenet", 
  "Facenet512", 
  "OpenFace", 
  "DeepFace", 
  "DeepID", 
  "ArcFace", 
  "Dlib", 
  "SFace",
  "GhostFaceNet",
]
dfs = DeepFace.find(
    img_path = "E:/lfw/lfw-py/lfw_funneled/Amer_al-Saadi/Amer_al-Saadi_0001.jpg",
    db_path = "E:/lfw/lfw-py/lfw_funneled",
)
# Get the file paths of matched faces
identity_list = dfs[0]["identity"].values

# Get the similarity scores/distances
distance_list = dfs[0]["distance"].values

# Combined display of matches with scores
for identity, distance in zip(identity_list, distance_list):
    print(f"Match: {identity}, Distance: {distance:.2f}")