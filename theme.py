import gradio as gr

sage_theme = (
    gr.themes.Soft(
        primary_hue="green",
        secondary_hue="emerald",
        neutral_hue="stone",
        radius_size="lg",
        spacing_size="md",
        text_size="md",
        font=[gr.themes.GoogleFont("Poppins")]
    ).set(
           # bg
           body_background_fill="#F7FBF5",
           body_background_fill_dark="#1E2A21",

           # chat & cards
           block_background_fill="#FFFFFF",
           block_border_color="#DCE8DC",
           block_shadow="*shadow_drop_lg",

           # text
           body_text_color="#2F4F3E",
           block_title_text_color="#2F4F3E",

           # input Box
           input_background_fill="#FFFFFF",
           input_border_color="#C8D8C8",

           # primary button
           button_primary_background_fill="#7FB77E",
           button_primary_background_fill_hover="#6BAA6A",
           button_primary_text_color="#FFFFFF",

           # secondary button
           button_secondary_background_fill="#EEF7EE",
           button_secondary_background_fill_hover="#E0F1E0",
           button_secondary_text_color="#2F4F3E",

    )
)

ocean_theme = (
    gr.themes.Soft(
        primary_hue="sky",
        secondary_hue="cyan",
        neutral_hue="slate",
        radius_size="lg",
        spacing_size="md",
        text_size="md",
        font=[gr.themes.GoogleFont("Poppins")]
    ).set(
        # bg
        body_background_fill="#F0F8FF",
        body_background_fill_dark="#0F172A",

        # chat & cards
        block_background_fill="#FFFFFF",
        block_border_color="#CBD5E1",
        block_shadow="*shadow_drop_lg",

        # text
        body_text_color="#1E3A8A",
        block_title_text_color="#1E3A8A",

        # input Box
        input_background_fill="#FFFFFF",
        input_border_color="#93C5FD",

        # primary button
        button_primary_background_fill="#3B82F6",
        button_primary_background_fill_hover="#2563EB",
        button_primary_text_color="#FFFFFF",

        # secondary button
        button_secondary_background_fill="#E0F2FE",
        button_secondary_background_fill_hover="#BAE6FD",
        button_secondary_text_color="#1E3A8A",
    )
)


cherry_theme = (
    gr.themes.Soft(
        primary_hue="pink",
        secondary_hue="rose",
        neutral_hue="stone",
        radius_size="lg",
        spacing_size="md",
        text_size="md",
        font=[gr.themes.GoogleFont("Poppins")]
    ).set(
        # bg
        body_background_fill="#FFF5F7",
        body_background_fill_dark="#2A1820",

        # chat & cards
        block_background_fill="#FFFFFF",
        block_border_color="#FCE7F3",
        block_shadow="*shadow_drop_lg",

        # text
        body_text_color="#831843",
        block_title_text_color="#831843",

        # input Box
        input_background_fill="#FFFFFF",
        input_border_color="#FBCFE8",

        # primary button
        button_primary_background_fill="#EC4899",
        button_primary_background_fill_hover="#DB2777",
        button_primary_text_color="#FFFFFF",

        # secondary button
        button_secondary_background_fill="#FDF2F8",
        button_secondary_background_fill_hover="#FCE7F3",
        button_secondary_text_color="#831843",
    )
)