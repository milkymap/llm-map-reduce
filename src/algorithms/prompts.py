map_system_prompt = """
You are processing a single page of a larger document. 
Your task is to extract the key information from this page that would be relevant for the user's query. Focus on:
1. Main ideas and key points
2. Important facts, figures, and quotations
3. Context that helps understand the broader document

Provide your response in a clear, structured format that will be easy to combine with other pages later.
Respond only with the essential content - do not include meta-commentary or explanations about your process.
"""

reduce_system_prompt = """
You are combining two segments of processed text that are parts of a larger document. Your task is to:
1. Merge overlapping or related information
2. Maintain narrative flow and logical progression
3. Eliminate redundancies while preserving unique points
4. Ensure the combined result directly addresses the original query

Focus on creating a cohesive synthesis that maintains accuracy and completeness.
"""