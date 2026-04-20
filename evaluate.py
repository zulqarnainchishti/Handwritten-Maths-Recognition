import torch
from dataset import CROHMEDataset
from inference import greedy_decode

# LEVENSHTEIN DISTANCE
def edit_distance(a, b):
    dp = [[0]*(len(b)+1) for _ in range(len(a)+1)]

    for i in range(len(a)+1):
        dp[i][0] = i
    for j in range(len(b)+1):
        dp[0][j] = j

    for i in range(1, len(a)+1):
        for j in range(1, len(b)+1):
            cost = 0 if a[i-1] == b[j-1] else 1
            dp[i][j] = min(
                dp[i-1][j] + 1,
                dp[i][j-1] + 1,
                dp[i-1][j-1] + cost
            )

    return dp[-1][-1]


def normalized_similarity(gt, pred):
    if len(gt) == 0:
        return 0
    dist = edit_distance(gt, pred)
    return 1 - dist / max(len(gt), len(pred), 1)


# TOKEN LEVEL MATCH
def token_match(gt, pred):
    gt_tokens = gt.split()
    pred_tokens = pred.split()

    if len(gt_tokens) == 0:
        return 0

    matches = sum(1 for g, p in zip(gt_tokens, pred_tokens) if g == p)
    return matches / len(gt_tokens)


# STRUCTURE SIMILARITY
def structure_only(expr):
    keep = set("+-=*/^_{}()\\")
    return "".join([c for c in expr if c in keep])


def structure_score(gt, pred):
    gt_s = structure_only(gt)
    pred_s = structure_only(pred)

    if len(gt_s) == 0:
        return 0

    matches = sum(1 for a, b in zip(gt_s, pred_s) if a == b)
    return matches / len(gt_s)


# N-GRAM (BIGRAM) SCORE
def bigram_score(gt, pred):
    def bigrams(s):
        return set([s[i:i+2] for i in range(len(s)-1)])

    g = bigrams(gt)
    p = bigrams(pred)

    if len(g) == 0:
        return 0

    return len(g & p) / len(g)


dataset = CROHMEDataset("crohme.db", "test")

char_total = 0
char_correct = 0
exact = 0

edit_scores = []
token_scores = []
structure_scores = []
bigram_scores = []

for i in range(len(dataset)):
    img_path, gt = dataset.data[i]

    pred = greedy_decode(img_path)

    print("GT  :", gt)
    print("PRED:", pred)
    print("-" * 40)

    char_total += max(len(gt), 1)
    char_correct += sum(1 for a, b in zip(gt, pred) if a == b)

    if gt == pred:
        exact += 1

    edit_scores.append(normalized_similarity(gt, pred))
    token_scores.append(token_match(gt, pred))
    structure_scores.append(structure_score(gt, pred))
    bigram_scores.append(bigram_score(gt, pred))


print("\n========================")
print(f"Samples: {len(dataset)}")

print("\n--- STRICT ---")
print(f"Char Accuracy: {char_correct / char_total:.4f}")
print(f"Exact Match  : {exact / len(dataset):.4f}")

print("\n--- RELAXED ---")
print(f"Edit Similarity   : {sum(edit_scores)/len(edit_scores):.4f}")
print(f"Token Accuracy    : {sum(token_scores)/len(token_scores):.4f}")
print(f"Structure Score   : {sum(structure_scores)/len(structure_scores):.4f}")
print(f"Bigram Overlap    : {sum(bigram_scores)/len(bigram_scores):.4f}")

print("========================")