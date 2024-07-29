custom_css_str = """
        <style>
            /* Main container styling */
            .main {
                padding: 2rem; /* Generous padding for a spacious feel */
                border-radius: 12px; /* Smooth, pronounced corners */
                box-shadow: 0 6px 12px rgba(0, 0, 0, 0.1); /* Enhanced shadow for modern effect */
                max-width: 100%; /* Ensure responsiveness */
                margin: 0 auto; /* Center align container */
                font-family: Arial, sans-serif; /* Modern font for consistency */
            }

            /* Responsive design for smaller screens */
            @media (max-width: 768px) {
                .main {
                    padding: 1rem; /* Reduced padding on smaller screens */
                }
            }

            /* Styling for Selectbox and Slider */
            .stSelectbox, .stSlider {
                border-radius: 8px; /* Consistent rounded corners */
                padding: 0.75rem; /* Adequate padding for better usability */
                margin-bottom: 1.5rem; /* Increased spacing between elements */
                border: 1px solid #e0e0e0; /* Subtle border */
                transition: border-color 0.3s ease, box-shadow 0.3s ease; /* Smooth transition for border and shadow */
            }

            /* Hover effects */
            .stSelectbox:hover, .stSlider:hover {
                border-color: #bdbdbd; /* Slightly darker border */
                box-shadow: 0 2px 6px rgba(0, 0, 0, 0.1); /* Subtle shadow effect */
            }

            /* Focus states */
            .stSelectbox:focus, .stSlider:focus {
                border-color: #007bff; /* Highlight border color */
                outline: none; /* Remove default outline */
                box-shadow: 0 0 0 3px rgba(0, 123, 255, 0.2); /* Focus shadow for accessibility */
            }

            /* Make sure elements are responsive */
            .stSelectbox select, .stSlider input {
                width: 100%; /* Full width for input elements */
            }

            /* Additional styling for sliders */
            .stSlider input[type="range"] {
                -webkit-appearance: none; /* Remove default appearance */
                width: 100%; /* Full width */
                height: 8px; /* Height of the slider */
                background: transparent; /* No background */
                border-radius: 4px; /* Rounded corners for the track */
                cursor: pointer; /* Pointer cursor */
                outline: none; /* Remove default outline */
                transition: background 0.3s ease; /* Smooth transition for background changes */
            }

            .stSlider input[type="range"]::-webkit-slider-thumb {
                -webkit-appearance: none; /* Remove default thumb appearance */
                width: 16px; /* Thumb width */
                height: 16px; /* Thumb height */
                border-radius: 50%; /* Rounded thumb */
                background: #007bff; /* Thumb color */
                cursor: pointer; /* Pointer cursor */
                transition: background 0.3s ease; /* Smooth transition for thumb changes */
            }

            .stSlider input[type="range"]::-moz-range-thumb {
                width: 16px; /* Thumb width */
                height: 16px; /* Thumb height */
                border-radius: 50%; /* Rounded thumb */
                background: #007bff; /* Thumb color */
                cursor: pointer; /* Pointer cursor */
                transition: background 0.3s ease; /* Smooth transition for thumb changes */
            }

            /* Additional styling for select boxes */
            .stSelectbox select {
                border: 1px solid #e0e0e0; /* Border color */
                padding: 0.5rem; /* Padding inside select */
                border-radius: 8px; /* Rounded corners */
                font-size: 1rem; /* Consistent font size */
                cursor: pointer; /* Pointer cursor */
                transition: border-color 0.3s ease; /* Smooth transition for border color */
            }
        </style>
    """
