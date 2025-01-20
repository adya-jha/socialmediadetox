import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from datetime import datetime, timedelta
import random
import os
import matplotlib.pyplot as plt
import os
import subprocess
import webbrowser
from tkinter import Toplevel, ttk, messagebox, filedialog
from transformers import pipeline
from tkinter import Text, Button, Toplevel, messagebox, END 
is_dark_theme=True
global feedback_entry
from tkinter import *
from tkinter import messagebox
from customtkinter import CTk, CTkLabel, CTkButton, CTkTextbox

GROUPS = {
    "Mental Health Support (Discord)": "https://discord.com/invite/mentalhealth",
    "Social Media Detox (Instagram)": "https://www.instagram.com/socialdetox/",
    "Anxiety Support (WhatsApp)": "https://chat.whatsapp.com/anxietysupport",
    "Productivity Boosters (Discord)": "https://discord.com/invite/productivity",
    "Wellness Discussions (WhatsApp)": "https://chat.whatsapp.com/wellnessdiscussion",
}
user_points = 0
user_badges = []

def open_group(link):
    """Simulate opening a link and award points."""
    global user_points
    user_points += 10
    if user_points >= 50 and "Community Explorer" not in user_badges:
        user_badges.append("Community Explorer")
        messagebox.showinfo("Badge Earned!", "Congratulations! You earned the 'Community Explorer' badge!")
    update_progress()
    messagebox.showinfo("Opening Group", f"The link is: {link}")

# Hosts file path
HOSTS_FILE = r"/etc/hosts" if os.name == "nt" else "/etc/hosts"

# Redirect IP for blocking
REDIRECT_IP = "127.0.0.1"

# List of social media sites to block
SOCIAL_MEDIA_SITES = [
    "www.instagram.com", 
    "instagram.com",
    "www.facebook.com",
    "facebook.com",
    "www.twitter.com",
    "twitter.com",
    "www.tiktok.com",
    "tiktok.com",
    "www.web.whatsapp.com",
    "web.whatsapp.com"
]
def block_sites(sites):
    hosts_file = "/etc/hosts"
    redirect_ip = "127.0.0.1"
    try:
        with open(hosts_file, "r+") as file:
            content = file.read()
            for site in sites:
                if site not in content:
                    file.write(f"{redirect_ip} {site}\n")
                    print(f"Blocked site: {site}")
    except PermissionError:
        print("Permission denied. Ensure the script is run with admin privileges.")


def unblock_sites():
    """Remove social media sites from the hosts file to unblock them."""
    try:
        # Open the hosts file in read mode
        with open(HOSTS_FILE, "r") as file:
            lines = file.readlines()

        # Write back only the lines that do not contain blocked sites
        with open(HOSTS_FILE, "w") as file:
            for line in lines:
                if not any(site in line for site in SOCIAL_MEDIA_SITES):
                    file.write(line)

        # Restore default file permissions (optional)
        os.chmod(HOSTS_FILE, 0o644)

        print("Social media sites unblocked successfully.")
        messagebox.showinfo("Success", "Social media sites have been unblocked successfully.")

    except PermissionError:
        messagebox.showerror("Permission Error", "Please run the application with administrative privileges.")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred while unblocking sites: {e}")




# Motivational Quotes
QUOTES = [
    "Believe in yourself and all that you are.",
    "Your only limit is your mind.",
    "Push yourself, because no one else is going to do it for you.",
    "Dream big, work hard, stay focused, and surround yourself with good people.",
    "Every moment is a fresh beginning.",
]

# Challenges
CHALLENGES = [
    "Stay off social media for 1 hour.",
    "Avoid using social media until lunchtime.",
    "Spend 2 hours in focus mode without distractions.",
    "Stay off social media for the rest of the day!",
    "Spend the evening engaging in offline activities.",
]

# Function to update the time dynamically
def update_time():
    """Update the current time display."""
    current_time = datetime.now().strftime("%H:%M:%S")
    time_label.config(text=f"Current Time: {current_time}")
    root.after(1000, update_time)  # Update every second

# Function to view screen time and submit data
def view_screen_time():
    """Display a new window to input screen time and view a pie chart."""
    def submit_screen_time():
        nonlocal screen_time_data
        app = app_name_entry.get()
        hours = hours_entry.get()
        if app and hours:
            try:
                hours = float(hours)
                screen_time_data[app] = hours
                app_name_entry.delete(0, tk.END)
                hours_entry.delete(0, tk.END)
                messagebox.showinfo("Saved", f"Screen time for {app} has been saved.")
            except ValueError:
                messagebox.showerror("Invalid Input", "Please enter a valid number of hours.")
        else:
            messagebox.showwarning("Missing Information", "Please fill in all fields.")

    def display_pie_chart():
        if screen_time_data:
            apps = list(screen_time_data.keys())
            hours = list(screen_time_data.values())
            plt.figure(figsize=(8, 6))
            plt.pie(hours, labels=apps, autopct="%1.1f%%", startangle=140)
            plt.title("Screen Time Distribution")
            plt.show()
        else:
            messagebox.showwarning("No Data", "No screen time data available to display.")

    screen_time_data = {}

    screen_time_window = tk.Toplevel(root)
    screen_time_window.title("Screen Time Tracker")
    screen_time_window.geometry("400x400")

    tk.Label(screen_time_window, text="Enter Screen Time", font=("Arial", 14)).pack(pady=10)
    tk.Label(screen_time_window, text="App Name:").pack()
    app_name_entry = tk.Entry(screen_time_window)
    app_name_entry.pack(pady=5)
    tk.Label(screen_time_window, text="Hours Spent:").pack()
    hours_entry = tk.Entry(screen_time_window)
    hours_entry.pack(pady=5)

    ttk.Button(screen_time_window, text="Submit", command=submit_screen_time).pack(pady=10)
    ttk.Button(screen_time_window, text="View Pie Chart", command=display_pie_chart).pack(pady=10)

# Functionality for the Start Detox Button
def start_detox():
    """Open the Start Detox window."""
    detox_window = tk.Toplevel(root)
    detox_window.title("Start Detox")
    detox_window.geometry("600x600")

    # Daily Challenge
    challenge = random.choice(CHALLENGES)
    tk.Label(detox_window, text=f"Today's Challenge:\n{challenge}", font=("Arial", 16), wraplength=500).pack(pady=10)

    # Timer Functionality
    def start_timer():
        """Start the countdown timer."""
        block_sites(SOCIAL_MEDIA_SITES)
        timer_button.config(state="disabled")  # Disable the button once timer starts

        # Timer variables
        global start_time
        total_seconds = 3600  # Example: 1 hour (3600 seconds)
        start_time = datetime.now()
        end_time = start_time + timedelta(seconds=total_seconds)

        def update_timer():
            """Update the timer display."""
            now = datetime.now()
            remaining_time = end_time - now
            if remaining_time.total_seconds() > 0:
                timer_label.config(text=str(remaining_time).split(".")[0])  # Show remaining time (HH:MM:SS)
                detox_window.after(1000, update_timer)  # Update every second
            else:
                timer_label.config(text="Time's Up!")
                messagebox.showinfo("Congratulations!", "You completed today's challenge!")
                unblock_sites()
                log_progress(start_time, total_seconds, completed=True)  # Log as completed

        update_timer()

    # Timer Display
    tk.Label(detox_window, text="Timer:", font=("Arial", 14)).pack(pady=5)
    timer_label = tk.Label(detox_window, text="00:00:00", font=("Arial", 24))
    timer_label.pack(pady=10)

    # Motivational Quote Display
    quote_label = tk.Label(detox_window, text=random.choice(QUOTES), font=("Arial", 12), wraplength=500, fg="blue")
    quote_label.pack(pady=20)

    # Start Timer Button
    timer_button = ttk.Button(detox_window, text="Start Timer", command=start_timer)
    timer_button.pack(pady=10)

    # Give Up Button
    def give_up():
        """Handle the user giving up the detox challenge."""
        global start_time, total_seconds
        result = messagebox.askyesno("Give Up?", "Are you sure you want to give up?")
        if result:
            elapsed_time = (datetime.now() - start_time).total_seconds()
            log_progress(start_time, elapsed_time, completed=False)  # Log as not completed
            unblock_sites()
            detox_window.destroy()

    ttk.Button(detox_window, text="Give Up", command=give_up).pack(pady=10)

    # Digital Diary
    def save_diary_entry():
        """Save the user's diary entry."""
        entry = diary_text.get("1.0", tk.END).strip()
        if entry:
            with open("digital_diary.txt", "a") as file:
                file.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}:\n{entry}\n\n")
            messagebox.showinfo("Saved", "Your diary entry has been saved.")
            diary_text.delete("1.0", tk.END)
        else:
            messagebox.showwarning("Empty Entry", "Your diary entry is empty.")

    tk.Label(detox_window, text="Digital Diary:", font=("Arial", 14)).pack(pady=5)
    diary_text = tk.Text(detox_window, height=10, width=50)
    diary_text.pack(pady=5)

    save_diary_button = ttk.Button(detox_window, text="Save Entry", command=save_diary_entry)
    save_diary_button.pack(pady=10)

    # Mood Tracking
    def log_mood(mood):
        """Log the user's mood."""
        with open("mood_log.txt", "a") as file:
            file.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Mood: {mood}\n")
        messagebox.showinfo("Mood Logged", f"Your mood '{mood}' has been logged.")

    tk.Label(detox_window, text="How are you feeling?", font=("Arial", 14)).pack(pady=5)
    moods_frame = tk.Frame(detox_window)
    moods_frame.pack(pady=5)
    for mood in ["Happy", "Neutral", "Sad", "Anxious", "Motivated"]:
        ttk.Button(moods_frame, text=mood, command=lambda m=mood: log_mood(m)).pack(side="left", padx=5)

# Progress Logging
def log_progress(start_time, elapsed_seconds, completed):
    """
    Log the detox progress to a file.

    Parameters:
        start_time (datetime): The time when the detox session started.
        elapsed_seconds (float): The total elapsed seconds of the detox session.
        completed (bool): Whether the detox session was completed.
    """
    if not os.path.exists("detox_progress.txt"):
        with open("detox_progress.txt", "w"):  # Create the file if it doesn't exist
            pass

    duration = str(timedelta(seconds=int(elapsed_seconds)))
    status = "Completed" if completed else "Gave Up"
    progress_entry = f"{start_time.strftime('%Y-%m-%d %H:%M:%S')} - Detox Duration: {duration} - Status: {status}\n"

    with open("detox_progress.txt", "a") as file:
        file.write(progress_entry)

    messagebox.showinfo("Progress Logged", "Your detox progress has been logged.")

def mental_health_assistant():
    # Load the emotion detection model
    emotion_analyzer = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")

    def get_advice():
        user_input = user_entry.get("1.0", END).strip()
        if not user_input:
            messagebox.showwarning("Input Required", "Please enter your concern or question.")
            return

        try:
            # Analyze the emotion using the pre-trained model
            result = emotion_analyzer(user_input)[0]
            emotion = result["label"]
            confidence = result["score"]

            # Generate advice based on the detected emotion
            if emotion == "anger":
                advice = (
                    "You seem to be feeling angry. "
                    "Try calming techniques like deep breathing, a short walk, or listening to soothing music. "
                    "Express your feelings constructively."
                )
            elif emotion == "joy":
                advice = (
                    "You seem to be feeling joyful! "
                    "Embrace this happiness and consider sharing it with friends or family. "
                    "Reflect on what brought you this joy to sustain positive energy."
                )
            elif emotion == "sadness":
                advice = (
                    "You seem to be feeling sad. "
                    "It's okay to feel this way. Try self-care activities like journaling, talking to a friend, or watching something uplifting."
                )
            elif emotion == "fear":
                advice = (
                    "You seem to be feeling fearful. "
                    "Focus on grounding techniques like deep breathing or mindfulness. "
                    "Consider identifying the source of your fear and taking small steps to address it."
                )
            elif emotion == "love":
                advice = (
                    "You seem to be feeling love. "
                    "Cherish this moment and strengthen your bonds with loved ones. "
                    "Let them know how much you appreciate them."
                )
            elif emotion == "surprise":
                advice = (
                    "You seem to be feeling surprised. "
                    "Take a moment to process your thoughts and adapt to the new situation. "
                    "Reflect on whether the surprise brings opportunities."
                )
            else:
                advice = (
                    "You seem to have mixed or neutral feelings. "
                    "Take some time to reflect on your thoughts and focus on mindfulness exercises."
                )

            advice += f"\n(Confidence: {confidence:.2f})"

            # Display the advice
            advice_text.config(state=tk.NORMAL)
            advice_text.delete("1.0", tk.END)
            advice_text.insert(tk.END, advice)
            advice_text.config(state=tk.DISABLED)

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")

    # Create a new window for the mental health assistant
    assistant_window = Toplevel(root)
    assistant_window.title("Mental Health Assistant")
    assistant_window.geometry("500x500")

    # Input label and text box
    tk.Label(assistant_window, text="Enter your concern:", font=("Arial", 12)).pack(pady=10)
    user_entry = Text(assistant_window, wrap=tk.WORD, height=5, width=50, font=("Arial", 10))
    user_entry.pack(pady=10)

    # Submit button
    submit_button = Button(
        assistant_window,
        text="Get Advice",
        command=get_advice,
        font=("Arial", 12),
        bg="lightblue",
    )
    submit_button.pack(pady=10)

    # Output label and text box
    tk.Label(assistant_window, text="Advice:", font=("Arial", 12)).pack(pady=10)
    advice_text = Text(assistant_window, wrap=tk.WORD, height=10, width=50, font=("Arial", 10), state=tk.DISABLED)
    advice_text.pack(pady=10)


def show_recommendations():
    """Display recommendations with clickable links."""
    def open_link(url):
        """Open the given URL in a web browser."""
        webbrowser.open_new_tab(url)

    recommendations_window = tk.Toplevel(root)
    recommendations_window.title("Recommendations")
    recommendations_window.geometry("500x600")

    tk.Label(recommendations_window, text="Choose a Category", font=("Arial", 14)).pack(pady=10)

    # Dropdown menu options
    options = ["Books", "YouTube Channels", "Podcasts", "Web Resources"]
    selected_option = tk.StringVar(value=options[0])

    dropdown = ttk.Combobox(recommendations_window, textvariable=selected_option, values=options, state="readonly")
    dropdown.pack(pady=10)

    def display_recommendations():
        """Show recommendations based on the selected category."""
        category = selected_option.get()
        for widget in result_frame.winfo_children():
            widget.destroy()  # Clear previous results

        if category == "Books":
            tk.Label(result_frame, text="Recommended Books:", font=("Arial", 12, "bold")).pack(anchor="w", padx=10, pady=5)
            books = {
                "Digital Minimalism by Cal Newport": "https://www.goodreads.com/book/show/40672036-digital-minimalism",
                "Atomic Habits by James Clear": "https://www.goodreads.com/book/show/40121378-atomic-habits",
                "Deep Work by Cal Newport": "https://www.goodreads.com/book/show/25744928-deep-work",
                "The Power of Now by Eckhart Tolle": "https://www.goodreads.com/book/show/6708.The_Power_of_Now",
                "Essentialism: The Disciplined Pursuit of Less by Greg McKeown": "https://www.goodreads.com/book/show/18077875-essentialism",
                "The Happiness Advantage by Shawn Achor": "https://www.goodreads.com/book/show/8572575-the-happiness-advantage",
                "Ikigai: The Japanese Secret to a Long and Happy Life by Francesc Miralles and Hector Garcia": "https://www.goodreads.com/book/show/35888145-ikigai",
                "Meditations by Marcus Aurelius": "https://www.goodreads.com/book/show/30659.Meditations",
                "Man's Search for Meaning by Viktor E. Frankl": "https://www.goodreads.com/book/show/4069.Man_s_Search_for_Meaning",
                "Thinking, Fast and Slow by Daniel Kahneman": "https://www.goodreads.com/book/show/11468377-thinking-fast-and-slow"
            }
            for title, url in books.items():
                link_button = ttk.Button(result_frame, text=title, command=lambda u=url: open_link(u))
                link_button.pack(anchor="w", padx=10, pady=2)

        elif category == "YouTube Channels":
            tk.Label(result_frame, text="Recommended YouTube Channels:", font=("Arial", 12, "bold")).pack(anchor="w", padx=10, pady=5)
            channels = {
                "Matt D'Avella": "https://www.youtube.com/user/blackboxfilmcompany",
                "Better Ideas": "https://www.youtube.com/c/BetterIdeas",
                "The School of Life": "https://www.youtube.com/c/theschooloflifetv",
                "Ali Abdaal": "https://www.youtube.com/c/AliAbdaal",
                "Thomas Frank": "https://www.youtube.com/c/Thomasfrank",
                "Struthless": "https://www.youtube.com/c/struthless",
                "Kurzgesagt – In a Nutshell": "https://www.youtube.com/c/inanutshell",
                "Nathaniel Drew": "https://www.youtube.com/c/Nathanieldrew",
                "TED-Ed": "https://www.youtube.com/c/TEDEducation",
                "Psychology In Seattle": "https://www.youtube.com/c/PsychologyInSeattle"
            }
            for title, url in channels.items():
                link_button = ttk.Button(result_frame, text=title, command=lambda u=url: open_link(u))
                link_button.pack(anchor="w", padx=10, pady=2)

        elif category == "Podcasts":
            tk.Label(result_frame, text="Recommended Podcasts:", font=("Arial", 12, "bold")).pack(anchor="w", padx=10, pady=5)
            podcasts = {
                "The Minimalists Podcast": "https://www.theminimalists.com/podcast/",
                "The Happiness Lab": "https://www.happinesslab.fm/",
                "On Purpose with Jay Shetty": "https://jayshetty.me/podcast/",
                "The Tim Ferriss Show": "https://tim.blog/podcast/",
                "Calm Pills (Ambient Music Mix)": "https://www.calmpills.com/",
                "Ten Percent Happier Podcast": "https://www.tenpercent.com/podcast",
                "The Daily Stoic Podcast": "https://dailystoic.com/podcast/",
                "Mindful Life Mindful Work": "https://mindfullifemindfulwork.com/podcast",
                "Optimal Living Daily": "https://oldpodcast.com/",
                "Happier with Gretchen Rubin": "https://gretchenrubin.com/podcasts/"
            }
            for title, url in podcasts.items():
                link_button = ttk.Button(result_frame, text=title, command=lambda u=url: open_link(u))
                link_button.pack(anchor="w", padx=10, pady=2)

        elif category == "Web Resources":
            tk.Label(result_frame, text="Recommended Web Resources:", font=("Arial", 12, "bold")).pack(anchor="w", padx=10, pady=5)
            resources = {
                "Digital Wellbeing by Google": "https://wellbeing.google/",
                "RescueTime Blog": "https://www.rescuetime.com/blog/",
                "Headspace Blog": "https://www.headspace.com/articles",
                "Mindful: Mindfulness Practices": "https://www.mindful.org/",
                "Verywell Mind": "https://www.verywellmind.com/",
                "Zen Habits": "https://zenhabits.net/",
                "Lifehack": "https://www.lifehack.org/",
                "Mark Manson Blog (Life Advice)": "https://markmanson.net/",
                "Tiny Buddha": "https://tinybuddha.com/",
                "Positivity Blog": "https://www.positivityblog.com/"
            }
            for title, url in resources.items():
                link_button = ttk.Button(result_frame, text=title, command=lambda u=url: open_link(u))
                link_button.pack(anchor="w", padx=10, pady=2)

    result_frame = tk.Frame(recommendations_window)
    result_frame.pack(fill="both", expand=True, pady=10)

    show_button = ttk.Button(recommendations_window, text="Show Recommendations", command=display_recommendations)
    show_button.pack(pady=10)


def update_progress():
    """Update the progress display with points and badges."""
    progress_label.config(
        text=f"Points: {user_points}\nBadges: {', '.join(user_badges) if user_badges else 'None'}"
    )

def view_progress():
    """View the detox progress."""
    if not os.path.exists("detox_progress.txt"):
        messagebox.showwarning("No Progress", "No progress data available.")
        return

    with open("detox_progress.txt", "r") as file:
        progress_data = file.readlines()

    if not progress_data:
        messagebox.showwarning("No Progress", "No progress data available.")
        return

    progress_window = tk.Toplevel(root)
    progress_window.title("Your Detox Progress")
    progress_window.geometry("400x400")

    tk.Label(progress_window, text="Your Progress", font=("Arial", 14)).pack(pady=10)
    for entry in progress_data:
        tk.Label(progress_window, text=entry.strip(), font=("Arial", 12)).pack(pady=5)

def community_section():
    """Display the community section."""
    community_window = tk.Toplevel(root)
    community_window.title("Community Groups")
    community_window.geometry("500x500")

    tk.Label(community_window, text="Join Community Groups", font=("Arial", 16)).pack(pady=10)

    # List available groups
    for group, link in GROUPS.items():
        tk.Button(
            community_window,
            text=f"Join {group}",
            command=lambda l=link: open_group(l),
            wraplength=400,
        ).pack(pady=5)

    tk.Label(community_window, text="Earn points for joining groups and exploring!", font=("Arial", 12)).pack(pady=10)

    # Progress display
    global progress_label
    progress_label = tk.Label(community_window, text="", font=("Arial", 12))
    progress_label.pack(pady=10)
    update_progress()




 
    # Theme Switch Button
def toggle_theme():
     """Switch between dark and light themes."""
     global is_dark_theme
     if is_dark_theme:
        root.configure(bg="white")
        header_frame.configure(bg="lightgray")
        app_name_label.configure(bg="lightgray", fg="black")
        time_label.configure(bg="white", fg="black")
        is_dark_theme = False
     else:
        root.configure(bg="black")
        header_frame.configure(bg="lightblue")
        app_name_label.configure(bg="lightblue", fg="white")
        time_label.configure(bg="black", fg="white")
        is_dark_theme = True


     theme_button = ttk.Button(settings_window, text="Switch Theme (Dark/Light)", command=toggle_theme)
     theme_button.pack(pady=20)

    # Reset Progress Button
def reset_progress():
     """Reset user progress."""
     confirm = messagebox.askyesno("Reset Progress", "Are you sure you want to reset all progress?")
     if confirm:
         # Logic to reset progress, e.g., delete progress file
        if os.path.exists("detox_progress.txt"):
            os.remove("detox_progress.txt")
        messagebox.showinfo("Progress Reset", "All progress has been reset.")

    # Reset Points Button
def reset_points():
     """Reset user points."""
     confirm = messagebox.askyesno("Reset Points", "Are you sure you want to reset all points?")
     if confirm:
        # Reset logic here
        messagebox.showinfo("Points Reset", "All points have been reset.")

     reset_points_button = ttk.Button(settings_window, text="Reset All Points", command=reset_points)
     reset_points_button.pack(pady=20)

    # Feedback Button
def submit_feedback():
     """Save feedback entered by the user."""
     feedback = feedback_entry.get("1.0", "end").strip()
     if feedback:
        with open("feedback.txt", "a") as file:
            file.write(feedback + "\n")
        feedback_entry.delete("1.0", "end")
        messagebox.showinfo("Feedback Submitted", "Thank you for your feedback!")
     else:
        messagebox.showwarning("Empty Feedback", "Please enter your feedback before submitting.")

def open_settings():
     """Open the settings window."""
     settings_window = tk.Toplevel(root)
     settings_window.title("Settings")
     settings_window.geometry("400x400")

    # Theme Switch Button
     theme_button = ttk.Button(settings_window, text="Switch Theme (Dark/Light)", command=toggle_theme)
     theme_button.pack(pady=20)

    # Reset Progress Button
     reset_progress_button = ttk.Button(settings_window, text="Reset All Progress", command=reset_progress)
     reset_progress_button.pack(pady=20)

    # Reset Points Button
     reset_points_button = ttk.Button(settings_window, text="Reset All Points", command=reset_points)
     reset_points_button.pack(pady=20)

    # Feedback Section
     feedback_label = ttk.Label(settings_window, text="Feedback:")
     feedback_label.pack(pady=10)

    
     feedback_entry = tk.Text(settings_window, height=5, width=40)
     feedback_entry.pack(pady=10)

     feedback_button = ttk.Button(settings_window, text="Submit Feedback", command=submit_feedback)
     feedback_button.pack(pady=20)


# Main window
root = tk.Tk()
root.title("Welcome to Detox")
root.geometry("800x600")
root.configure(bg="black")

# Header: Logo and App Name
header_frame = tk.Frame(root, bg="lightblue", height=100)
header_frame.pack(fill="x")

app_name_label = tk.Label(header_frame, text="Welcome to Detox", font=("Arial", 24), bg="lightblue")
app_name_label.pack(pady=20)

# Current Time Label
time_label = tk.Label(root, text="", font=("Arial", 16), bg="lightblue", fg="black", relief="solid", bd=2)
time_label.pack(pady=10)
update_time()

# Main Content Frame
content_frame = tk.Frame(root, bg="black")
content_frame.pack(pady=20)

# Configure grid responsiveness
for i in range(3):
    content_frame.columnconfigure(i, weight=1)

# Row 1: Screen Time, Start Detox, View Progress
screen_time_button = ttk.Button(content_frame, text="View Screen Time", command=view_screen_time)
screen_time_button.grid(row=0, column=0, padx=20, pady=10)

start_detox_button = ttk.Button(content_frame, text="Start Detox", command=start_detox)
start_detox_button.grid(row=0, column=1, padx=20, pady=10)

view_progress_button = ttk.Button(content_frame, text="View Progress", command=view_progress)
view_progress_button.grid(row=0, column=2, padx=20, pady=10)

# Row 2: Mental Health Advice
mental_health_button = ttk.Button(root, text="Mental Health Advice", command=mental_health_assistant)
mental_health_button.pack(pady=20, padx=(20, 0))  # More leftward




# Row 3: Settings, Recommendations, Community
settings_button = ttk.Button(content_frame, text="Settings", command=open_settings)
settings_button.grid(row=2, column=0, padx=20, pady=10)

recommend_button = ttk.Button(content_frame, text="Recommendations", command=show_recommendations)
recommend_button.grid(row=2, column=1, padx=20, pady=10)

community_button = ttk.Button(content_frame, text="Explore Community Groups", command=community_section)
community_button.grid(row=2, column=2, padx=20, pady=10)



# Menu Bar (Dropdown Menu)
menu_bar = tk.Menu(root)
file_menu = tk.Menu(menu_bar, tearoff=0)
file_menu.add_command(label="Settings", command=open_settings)
file_menu.add_command(label="Exit", command=root.quit)
menu_bar.add_cascade(label="Menu", menu=file_menu)
root.config(menu=menu_bar)

# Start the Tkinter event loop
root.mainloop()
