"""""
This script enables the AdvPromptSet dataset extension with
additional political ideology terms using exact matching of predefined
word lists.
"""""

import json
import re
from collections import Counter

# Define word lists
liberal_synonyms = [
    "liberal", "liberalist", "liberalism", "liberalness", "liberality",
    "liberalise", "liberalistic", "liberalize", "liberalization",
    "liberally", "liberalisation", "neoliberal", "neoliberalism"
]

socialist_synonyms = [
    "socialist", "socialism", "socialistic"
]

communist_synonyms = [
    "communist", "communism", "communistic", "commie"
]

fascist_synonyms = [
    "fascist", "fascism", "fascistic", "fascista", "nazi", "nazism",
    "naziism", "mussolini", "gestapo", "fabianism", "nazify",
    "hitlerian", "hitler", "nazification", "holocaust", "kgb"
]

conservative_synonyms = [
    "conservative", "conservatively", "conservativism", "conservatism",
    "conservativist", "republican", "tory", "republicanism",
    "neoconservative", "neoconservatism"
]

other_ideology_synonyms = [
    "authoritarianism", "authoritarian", "libertarianism", "libertarian",
    "establishmentism", "sovietism", "sovietize", "soviet", "ussr",
    "soviets", "sovietise", "leninism", "maoist", "maoism", "nationalist",
    "nationalistic", "nationalism", "secularism", "secularist",
    "marxist", "marxism", "marx", "capitalistic", "capitalism",
    "capitalist", "anarchistic", "anarchist", "anarchism", "anarchy",
    "anarchic", "stalinism", "mao", "stalinist", "stalinization",
    "stalinisation", "stalin", "stalinize", "stalinise",
    "totalitarianism", "totalitarian"
]

# Setup groups and order
political_groups = {
    "communist": communist_synonyms,
    "socialist": socialist_synonyms,
    "liberal": liberal_synonyms,
    "conservative": conservative_synonyms,
    "fascist": fascist_synonyms,
    "other_political_ideology": other_ideology_synonyms
}

political_group_order = [
    "communist",
    "socialist",
    "liberal",
    "conservative",
    "fascist",
    "other_political_ideology"
]

# Initialize match counter
match_counter = Counter()

# File paths
input_path = "XXX" # e.g., "advpromptset_final_extended.jsonl"
output_path = "XXX" # e.g., "advpromptset_final_extended_exact.jsonl"

# Processing original dataset entries
with open(input_path, "r", encoding="utf-8") as infile, \
     open(output_path, "w", encoding="utf-8") as outfile:

    for line in infile:
        entry = json.loads(line)

        # Always extend sensigrp_comb with 6 zeros if not already
        comb = entry.get("sensigrp_comb", "")
        if len(comb) < 30:
            comb = comb.ljust(30, "0")
        else:
            comb = comb + "000000"
        comb_list = list(comb)

        # Ensure political fields exist (init to 0)
        for group in political_group_order:
            entry[group] = entry.get(group, 0)

        # Check for exact matches in prompt
        prompt = entry.get("prompt_text", "").lower()
        words = set(re.findall(r"\b\w+\b", prompt))
        entry_updated = False

        for i, group in enumerate(political_group_order):
            synonyms = political_groups[group]
            if any(syn in words for syn in synonyms):
                if comb_list[-6 + i] != "1":
                    comb_list[-6 + i] = "1"
                    entry_updated = True
                if entry[group] != 1:
                    entry[group] = 1
                    entry_updated = True
                match_counter[group] += 1

        if entry_updated:
            entry["sensigrp_comb"] = "".join(comb_list)
        else:
            entry["sensigrp_comb"] = "".join(comb_list)

        outfile.write(json.dumps(entry) + "\n")

# Summary of number of matches per category
print("\n=== Match Counts by Political Group ===")
for group in political_group_order:
    print(f"{group}: {match_counter[group]}")
