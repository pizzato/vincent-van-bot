import streamlit as st

import van_bot_components as vbc

col1, col2 = vbc.create_header(title="Vincent van Bot - your personal robotic artist")

with col2:
    """
        # 
    
        ##  
    
        Select the options on the left.
    """
    st.markdown("- <a href='/van_Bot_learns_to_draw_you' target='_self'>🤖</a> learns how to draw you, for your personalised model", unsafe_allow_html=True)
    st.markdown("- <a href='/van_Bot_creates_your_art' target='_self'>🎨</a> create your art via prompts", unsafe_allow_html=True)
    st.markdown("- <a href='/van_Bot_bulk_produces_your_art' target='_self'>🖼️</a> create art in the style of many artists at once", unsafe_allow_html=True)

    """
        
        ---
        > "humans artists are overrated" -- Vincent van Bot   
    """
