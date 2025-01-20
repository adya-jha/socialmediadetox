**OVERVIEW**

The Social Media Detox application is a user-friendly tool designed to help individuals reduce their screen time
and improve productivity. With an intuitive interface and AI-driven features, this application guides users through 
the detox process while providing insights and motivation to stay on track.

**features**

1)Screen Time Analysis:
View your daily screen time in a pie chart to understand usage patterns.
Analyze time spent on different activities and platforms.

2)Detox Mode:
Block access to social media URLs for a designated time (e.g., one hour).
Utilize the Digital Entry feature to jot down your thoughts during the detox.

3)Progress Tracking:
Monitor your progress through detailed visuals and stats.
Celebrate milestones in reducing screen time.

4)Personalized Recommendations:
Access links to curated books, YouTube channels, podcasts, and more.
Directly navigate to recommended resources for self-improvement and relaxation.

5)Community Support:
Join communities on Discord, WhatsApp, Instagram, etc., to connect with like-minded individuals.
Share your journey and gain motivation from others.

6)Mental Health Chatbot:
Get instant mental health advice through a chatbot.
Receive tips and techniques to manage anxiety, sadness, and other emotions.

**Technologies and Libraries Used**

Programming Language: Python

GUI Frameworks:
tkinter and customtkinter for building the user interface.

Visualization:
matplotlib for creating pie charts and progress visualizations.

AI and NLP:
transformers library for building the mental health chatbot.

**Utilities:**

datetime for managing schedules and progress tracking.

webbrowser for opening recommended resources directly.

random, os, and subprocess for additional system functionalities.

**SPECIAL NOTE** 
*(Managing System Privileges)*

The application modifies the /etc/hosts file to block social media URLs during detox sessions.
To ensure your system's integrity, follow these steps:

Grant Write Permission to /etc/hosts: 

Run the following command to temporarily allow the application to modify the hosts file:

sudo chmod 777 /etc/hosts

Run the Application: Launch the application as usual:

python social.py

Revert Permissions: Immediately after running the application, restore the default permissions for the hosts file to protect your system:

sudo chmod 644 /etc/hosts

This ensures the file is writable only by the root user and readable by everyone.









**Contact**

Author: Adya Jha

Email: adyajha498@gmail.com

GitHub: adya-jha

LinkedIn: Adya Jha


