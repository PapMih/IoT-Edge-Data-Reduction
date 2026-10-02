import pandas as pd

# 1. Φόρτωσε το αρχείο σου
df = pd.read_csv("DATASET_CMA_CGM_NERVAL_5min.csv")

# 2. Κράτα μόνο τις βασικές στήλες που μας νοιάζουν (άλλαξέ τες αν θες)
cols = ['SPEEDKNOTS', 'MEPOWER', 'MEFOFLOW', 'MERPM']
df_check = df[cols].copy()

# 3. Πέταξε τα "σκουπίδια" (ΠΡΕΠΕΙ ΟΛΑ να ισχύουν ταυτόχρονα)
# Προσαρμόζεις τα όρια βάσει αυτών που σου είπε ο καθηγητής
valid_data = df_check[
    (df_check['SPEEDKNOTS'] > 5) & (df_check['SPEEDKNOTS'] < 25) &
    (df_check['MEPOWER'] > 1000) &  # Βάλε ένα λογικό ελάχιστο
    (df_check['MEFOFLOW'] > 0.5) &
    (df_check['MERPM'] > 30)
].copy()

# 4. Ψάχνουμε για ΣΥΝΕΧΟΜΕΝΕΣ εγγραφές (νησίδα)
# Υπολογίζουμε τη διαφορά στα index. Αν δεν είναι 1, το σερί έσπασε.
valid_data['block'] = (valid_data.index.to_series().diff() != 1).cumsum()

# Βρες το μεγαλύτερο συνεχόμενο μπλοκ
biggest_block_id = valid_data['block'].value_counts().idxmax()
biggest_block_size = valid_data['block'].value_counts().max()

print(f"Η μεγαλύτερη συνεχόμενη 'καθαρή' νησίδα έχει: {biggest_block_size} εγγραφές.")

if biggest_block_size >= 300:
    print("\nΣΩΘΗΚΑΜΕ! Έχουμε αρκετά δεδομένα για τον πίνακα συνδιακύμανσης.")
    clean_island = valid_data[valid_data['block'] == biggest_block_id]
    Sigma = clean_island[cols].cov()
    print("\nΟ Πίνακας Συνδιακύμανσης (Sigma) είναι:")
    print(Sigma)
else:
    print("\nΔΥΣΤΥΧΩΣ τα δεδομένα είναι πολύ σπασμένα. Πάμε αναγκαστικά στη Μέθοδο 2 (Χειροκίνητες εξισώσεις).")