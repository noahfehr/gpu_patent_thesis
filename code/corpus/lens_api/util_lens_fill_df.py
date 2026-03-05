# This function will take the JSON output from the lens API, and a master pandas df
# It then fills in the information that is missing from the master df
# Error handeling will need to be implemented to ensure that the function does not break when data is missing
import pandas as pd
import json
import os


#def lens_fill_df(lens_output, master_df):
#def lens_fill_df(path_input, path_output):
def lens_fill_df(path_input:str, path_output: str) -> None:
    """
    Processes JSON output from the Lens API and saves a pandas DataFrame with selected patent information.

    Parameters:
        path_input (str): Path to the JSON file containing Lens API output.
        path_output (str): Path to the CSV file where the DataFrame will be saved.

    The DataFrame includes:
        - lens_id: Unique identifier for the patent.
        - jurisdiction: Patent jurisdiction.
        - doc_number: Document number.
        - kind_code: Kind code of the patent.
        - publication_type: Type of publication.
        - simp_famil_lens_ids: List of lens_ids in the same simple family.
        - for_cite_lens_ids: List of lens_ids citing this patent.
        - back_cite_lens_ids: List of lens_ids cited by this patent.
        - priority_date: Priority date of the patent.
        - publish_date: Publication date of the patent.
        - cpc_codes: List of CPC classification codes.
        - claims: List of patent claims.
        - title: Patent title.
        - collection_file: Source filename (without extension).
    """
    with open(path_input, "r") as file:
        data = json.load(file)
    #print (data[7])
    #print(type(data))
    records = []

    filename = os.path.splitext(os.path.basename(path_input))[0]  # save filename from which data is extracted without extension

    for index, patents in enumerate(data):
        lens_id = str(patents.get("lens_id", None)) #to have everything in the dataframe as a string type
        jurisdiction = str(patents.get("jurisdiction", None))
        publication_type = str(patents.get("publication_type", None))
        kind_code = str(patents.get("kind", None))
        doc_number = str(patents.get("doc_number", None))

        #simp_famil_id = str(patents.get("docdb_id", None))
        for_cite_lens_ids = [str(pat.get("lens_id")) for pat in patents.get("biblio", {}).get("cited_by", {}).get("patents", []) if pat]   #could add a control if #string == patent_count (but maybe not necessary)
        back_cite_lens_ids = [str(cit.get("patcit", {}).get("lens_id")) for cit in patents.get("biblio", {}).get("references_cited", {}).get("citations", []) if cit and cit.get("patcit")]
        simp_famil_lens_ids = [str(member.get("lens_id")) for member in patents.get("families", {}).get("simple_family", {}).get("members", []) if member]

        # priority_date =  str(patents.get("biblio", {}).get("application_reference", []).get("date", None))
        priority_date =  str(patents.get("biblio", {}).get("priority_claims", {}).get("earliest_claim", {}).get("date", None)) # the priority date is actually given by the earliest priority claim date
        priority_claims = patents.get("biblio", {}).get("priority_claims", {})
        priority_jurisdiction = str(next(
            (claim.get("jurisdiction") for claim in priority_claims.get("claims", []) if claim.get("date") == priority_date),
            None
        ))

    
        publish_date = str(patents.get("date_published", None))
        applicant = [str(applicants.get("extracted_name", {}).get("value", {})) for applicants in patents.get("biblio", {}).get("parties", {}).get("applicants", []) if applicants]
        inventor_jurisdiction = [str(inv.get("residence")) for inv in patents.get("biblio", {}).get("parties", {}).get("inventors", []) if inv.get("residence") is not None]

        cpc_codes = [str(cpc.get("symbol")) for cpc in patents.get("biblio", {}).get("classifications_cpc", {}).get("classifications", {}) if cpc]
        title = str(patents.get("biblio", {}).get("invention_title", [{}])[0].get("text", {}))
        
        # Improved claims extraction with better error handling
        claims = []
        claims_data = patents.get("claims", [])
        if claims_data and isinstance(claims_data, list) and len(claims_data) > 0:
            claims_dict = claims_data[0] if claims_data[0] else {}
            claims_list = claims_dict.get("claims", [])
            for claim in claims_list:
                if isinstance(claim, dict) and "claim_text" in claim:
                    claim_texts = claim.get("claim_text", [])
                    if isinstance(claim_texts, list):
                        claims.extend([str(text) for text in claim_texts if text])
                    elif claim_texts:  # Single claim text as string
                        claims.append(str(claim_texts))
        
        abstract = str(patents.get("abstract", [{}])[0].get("text", {}))
        #description = str(patents.get("description", None))

#         "include": ["lens_id", "jurisdiction", "doc_number", "kind", "date_published","publication_type", "biblio", "families", "claims"]
#          Get jurisdiction, publication type

        if not back_cite_lens_ids:
            back_cite_lens_ids = []
        if not for_cite_lens_ids:
            for_cite_lens_ids = []
        if not simp_famil_lens_ids:
            simp_famil_lens_ids = []

        records.append({
            "lens_id": lens_id,
            "jurisdiction": jurisdiction,
            "doc_number":doc_number,
            "kind_code": kind_code,
            "publication_type": publication_type,
            "simp_famil_lens_ids": simp_famil_lens_ids,
            "for_cite_lens_ids": for_cite_lens_ids,
            "back_cite_lens_ids": back_cite_lens_ids,
            "priority_date": priority_date,
            "priority_jurisdiction": priority_jurisdiction,
            "publish_date": publish_date,
            "cpc_codes": cpc_codes,
            "claims": claims,
            "title": title,
            "applicant": applicant,
            "inventor_jurisdiction": inventor_jurisdiction,
            "abstract": abstract,
            "collection_file": filename
#            "description": description
        })


    df = pd.DataFrame(records)  #create the df after the loop when all the datas are already added in a list is a bit more efficient because you are not continuously updating df inside the loop

    if os.path.exists(path_output):
        df_old = pd.read_csv(path_output)
        df = pd.concat([df_old, df], ignore_index=True)

    df.to_csv(path_output, index= False)  #it could be good to add all the input/output path in args_init.json, but maybe in the end when you decide how many file (and from which function) you want to save
    print(f"Filled DF has been successfully saved to {path_output}.")
    print(f'Total number of entries in the df:{df.shape[0]}')

    #Check duplicate entries
    ids_not_dupl = list(set(df['lens_id']))
    print(f'Total number of unique lens_ids data collected for {len(ids_not_dupl)}')
    return


