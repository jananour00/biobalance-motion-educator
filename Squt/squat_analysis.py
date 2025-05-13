import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
import csv
from datetime import datetime
import math


class PoseAnalyzer:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLOv8 Pose Analysis Tool")
        self.root.geometry("600x400")
        self.root.configure(bg="#f0f0f0")

        # Create style
        self.style = ttk.Style()
        self.style.configure("TFrame", background="#f0f0f0")
        self.style.configure("TButton", font=("Arial", 10, "bold"))
        self.style.configure("TLabel", background="#f0f0f0", font=("Arial", 10))

        # Initialize variables - Make sure these are defined BEFORE setting up tabs
        self.video_path = ""
        self.export_path = os.path.join(os.path.expanduser("~"), "Documents")
        self.results_data = {
            "timestamps": [],
            "elbow_angles": [],
            "knee_angles": [],
            "hip_angles": [],
            "custom_y": []
        }
        self.skip_frames = 1
        self.video_scale = 1.0
        self.confidence_threshold = 0.5

        # Visual style settings for angle display
        self.angle_line_thickness = 3
        self.angle_font_scale = 0.8
        self.angle_font_thickness = 2
        self.angle_colors = {
            "elbow": (255, 0, 0),  # Blue (BGR)
            "knee": (0, 0, 255),  # Red (BGR)
            "hip": (0, 255, 255)  # Yellow (BGR)
        }

        # Load model
        try:
            self.model = YOLO("yolov8n-pose.pt")
        except Exception as e:
            messagebox.showerror("Model Error", f"Failed to load YOLO model: {str(e)}")
            root.destroy()
            return

        # Keypoint mapping
        self.keypoint_names = [
            "Nose", "Left Eye", "Right Eye", "Left Ear", "Right Ear",
            "Left Shoulder", "Right Shoulder", "Left Elbow", "Right Elbow",
            "Left Wrist", "Right Wrist", "Left Hip", "Right Hip",
            "Left Knee", "Right Knee", "Left Ankle", "Right Ankle"
        ]

        # Create main frames
        self.main_frame = ttk.Frame(root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Setup tabs
        self.tab_control = ttk.Notebook(self.main_frame)

        self.analysis_tab = ttk.Frame(self.tab_control)
        self.settings_tab = ttk.Frame(self.tab_control)
        self.visual_tab = ttk.Frame(self.tab_control)  # New tab for visual settings

        self.tab_control.add(self.analysis_tab, text="Analysis")
        self.tab_control.add(self.settings_tab, text="Settings")
        self.tab_control.add(self.visual_tab, text="Visual Settings")  # Added new tab
        self.tab_control.pack(expand=1, fill=tk.BOTH)

        # --- Analysis Tab ---
        self.setup_analysis_tab()

        # --- Settings Tab ---
        self.setup_settings_tab()

        # --- Visual Settings Tab ---
        self.setup_visual_tab()

    def setup_analysis_tab(self):
        # Frame for controls
        control_frame = ttk.Frame(self.analysis_tab)
        control_frame.pack(fill=tk.X, pady=10)

        # Video selection
        ttk.Label(control_frame, text="Video File:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.video_path_var = tk.StringVar()
        ttk.Entry(control_frame, textvariable=self.video_path_var, width=40).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(control_frame, text="Browse", command=self.select_video).grid(row=0, column=2, padx=5, pady=5)

        # Custom point tracking
        ttk.Label(control_frame, text="Track Point:").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        self.custom_point_var = tk.StringVar()
        self.combo = ttk.Combobox(control_frame, textvariable=self.custom_point_var, values=self.keypoint_names,
                                  state="readonly", width=15)
        self.combo.current(5)  # Default to Left Shoulder
        self.combo.grid(row=1, column=1, padx=5, pady=5, sticky=tk.W)

        # Joint selection
        self.joint_frame = ttk.LabelFrame(self.analysis_tab, text="Joints to Analyze")
        self.joint_frame.pack(fill=tk.X, pady=10, padx=10)

        self.track_elbow = tk.BooleanVar(value=True)
        self.track_knee = tk.BooleanVar(value=True)
        self.track_hip = tk.BooleanVar(value=True)  # Changed default to True

        ttk.Checkbutton(self.joint_frame, text="Elbow Angle", variable=self.track_elbow).grid(row=0, column=0, padx=20,
                                                                                              pady=5, sticky=tk.W)
        ttk.Checkbutton(self.joint_frame, text="Knee Angle", variable=self.track_knee).grid(row=0, column=1, padx=20,
                                                                                            pady=5, sticky=tk.W)
        ttk.Checkbutton(self.joint_frame, text="Hip Angle", variable=self.track_hip).grid(row=0, column=2, padx=20,
                                                                                          pady=5, sticky=tk.W)

        # Side selection
        self.side_var = tk.StringVar(value="left")
        ttk.Radiobutton(self.joint_frame, text="Left Side", variable=self.side_var, value="left").grid(row=1, column=0,
                                                                                                       padx=20, pady=5,
                                                                                                       sticky=tk.W)
        ttk.Radiobutton(self.joint_frame, text="Right Side", variable=self.side_var, value="right").grid(row=1,
                                                                                                         column=1,
                                                                                                         padx=20,
                                                                                                         pady=5,
                                                                                                         sticky=tk.W)

        # Button frame
        button_frame = ttk.Frame(self.analysis_tab)
        button_frame.pack(fill=tk.X, pady=20)

        # Run button
        run_button = ttk.Button(button_frame, text="Run Analysis", command=self.run_analysis, style="TButton")
        run_button.pack(side=tk.LEFT, padx=10)

        # Export button
        export_button = ttk.Button(button_frame, text="Export Data", command=self.export_data)
        export_button.pack(side=tk.LEFT, padx=10)

        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(self.analysis_tab, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def setup_settings_tab(self):
        settings_frame = ttk.Frame(self.settings_tab, padding=20)
        settings_frame.pack(fill=tk.BOTH, expand=True)

        # Video processing settings
        ttk.Label(settings_frame, text="Skip Frames:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.skip_frames_var = tk.IntVar(value=self.skip_frames)  # Use the initialized value
        skip_spin = ttk.Spinbox(settings_frame, from_=1, to=10, textvariable=self.skip_frames_var, width=5)
        skip_spin.grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
        ttk.Label(settings_frame, text="(Higher values = faster processing, lower resolution)").grid(row=0, column=2,
                                                                                                     sticky=tk.W,
                                                                                                     padx=5, pady=5)

        # Video display scale
        ttk.Label(settings_frame, text="Display Scale:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.scale_var = tk.DoubleVar(value=self.video_scale)  # Use the initialized value
        scale_combo = ttk.Combobox(settings_frame, textvariable=self.scale_var,
                                   values=[0.25, 0.5, 0.75, 1.0, 1.25, 1.5], width=5)
        scale_combo.grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)

        # Confidence threshold
        ttk.Label(settings_frame, text="Confidence Threshold:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        self.conf_var = tk.DoubleVar(value=self.confidence_threshold)  # Use the initialized value
        conf_scale = ttk.Scale(settings_frame, from_=0.1, to=0.9, orient=tk.HORIZONTAL,
                               variable=self.conf_var, length=200)
        conf_scale.grid(row=2, column=1, columnspan=2, sticky=tk.W, padx=5, pady=5)
        ttk.Label(settings_frame, textvariable=self.conf_var).grid(row=2, column=3, sticky=tk.W)

        # Export path
        ttk.Label(settings_frame, text="Export Directory:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=5)
        self.export_path_var = tk.StringVar(value=self.export_path)  # Use the initialized value
        ttk.Entry(settings_frame, textvariable=self.export_path_var, width=40).grid(row=3, column=1, columnspan=2,
                                                                                    sticky=tk.W, padx=5, pady=5)
        ttk.Button(settings_frame, text="Browse", command=self.select_export_path).grid(row=3, column=3, padx=5, pady=5)

        # Apply button
        ttk.Button(settings_frame, text="Apply Settings", command=self.apply_settings).grid(row=4, column=1, pady=20)

    def setup_visual_tab(self):
        visual_frame = ttk.Frame(self.visual_tab, padding=20)
        visual_frame.pack(fill=tk.BOTH, expand=True)

        # Line thickness settings
        ttk.Label(visual_frame, text="Line Thickness:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.line_thickness_var = tk.IntVar(value=self.angle_line_thickness)
        thickness_spin = ttk.Spinbox(visual_frame, from_=1, to=10, textvariable=self.line_thickness_var, width=5)
        thickness_spin.grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)

        # Font scale
        ttk.Label(visual_frame, text="Font Scale:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.font_scale_var = tk.DoubleVar(value=self.angle_font_scale)
        font_scale_spin = ttk.Spinbox(visual_frame, from_=0.5, to=2.0, increment=0.1, textvariable=self.font_scale_var,
                                      width=5)
        font_scale_spin.grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)

        # Font thickness
        ttk.Label(visual_frame, text="Font Thickness:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        self.font_thickness_var = tk.IntVar(value=self.angle_font_thickness)
        font_thickness_spin = ttk.Spinbox(visual_frame, from_=1, to=5, textvariable=self.font_thickness_var, width=5)
        font_thickness_spin.grid(row=2, column=1, sticky=tk.W, padx=5, pady=5)

        # Color selection frame
        color_frame = ttk.LabelFrame(visual_frame, text="Angle Colors")
        color_frame.grid(row=3, column=0, columnspan=4, sticky=tk.W, padx=5, pady=10)

        # Elbow color
        ttk.Label(color_frame, text="Elbow:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.elbow_color_var = tk.StringVar(value="Blue")
        elbow_combo = ttk.Combobox(color_frame, textvariable=self.elbow_color_var,
                                   values=["Blue", "Red", "Green", "Yellow", "Cyan", "Magenta"], width=10)
        elbow_combo.grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)

        # Knee color
        ttk.Label(color_frame, text="Knee:").grid(row=0, column=2, sticky=tk.W, padx=5, pady=5)
        self.knee_color_var = tk.StringVar(value="Red")
        knee_combo = ttk.Combobox(color_frame, textvariable=self.knee_color_var,
                                  values=["Blue", "Red", "Green", "Yellow", "Cyan", "Magenta"], width=10)
        knee_combo.grid(row=0, column=3, sticky=tk.W, padx=5, pady=5)

        # Hip color
        ttk.Label(color_frame, text="Hip:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.hip_color_var = tk.StringVar(value="Yellow")
        hip_combo = ttk.Combobox(color_frame, textvariable=self.hip_color_var,
                                 values=["Blue", "Red", "Green", "Yellow", "Cyan", "Magenta"], width=10)
        hip_combo.grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)

        # Display arc option
        self.show_arc_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(visual_frame, text="Show Angle Arc", variable=self.show_arc_var).grid(
            row=4, column=0, columnspan=2, sticky=tk.W, padx=5, pady=5)

        # Apply button
        ttk.Button(visual_frame, text="Apply Visual Settings", command=self.apply_visual_settings).grid(
            row=5, column=0, columnspan=2, pady=20)

    def select_video(self):
        path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv")])
        if path:
            self.video_path = path
            self.video_path_var.set(path)

    def select_export_path(self):
        path = filedialog.askdirectory()
        if path:
            self.export_path = path
            self.export_path_var.set(path)

    def apply_settings(self):
        try:
            self.skip_frames = self.skip_frames_var.get()
            self.video_scale = self.scale_var.get()
            self.confidence_threshold = self.conf_var.get()
            self.export_path = self.export_path_var.get()
            messagebox.showinfo("Settings", "Settings applied successfully")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to apply settings: {str(e)}")

    def apply_visual_settings(self):
        try:
            self.angle_line_thickness = self.line_thickness_var.get()
            self.angle_font_scale = self.font_scale_var.get()
            self.angle_font_thickness = self.font_thickness_var.get()

            # Map color names to BGR values
            color_map = {
                "Blue": (255, 0, 0),  # BGR format
                "Red": (0, 0, 255),
                "Green": (0, 255, 0),
                "Yellow": (0, 255, 255),
                "Cyan": (255, 255, 0),
                "Magenta": (255, 0, 255)
            }

            self.angle_colors = {
                "elbow": color_map[self.elbow_color_var.get()],
                "knee": color_map[self.knee_color_var.get()],
                "hip": color_map[self.hip_color_var.get()]
            }

            messagebox.showinfo("Visual Settings", "Visual settings applied successfully")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to apply visual settings: {str(e)}")

    def calculate_angle(self, a, b, c):
        a, b, c = np.array(a), np.array(b), np.array(c)
        ba = a - b
        bc = c - b
        cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        angle = np.arccos(np.clip(cos_angle, -1, 1))
        return np.degrees(angle)

    def draw_angle(self, frame, point1, point2, point3, angle, color, label=""):
        """Draw angle visualization with lines, arc and text"""
        # Convert points to integers for drawing
        p1 = (int(point1[0]), int(point1[1]))
        p2 = (int(point2[0]), int(point2[1]))
        p3 = (int(point3[0]), int(point3[1]))

        # Draw lines
        cv2.line(frame, p1, p2, color, self.angle_line_thickness)
        cv2.line(frame, p3, p2, color, self.angle_line_thickness)

        # Calculate vectors and normalized versions for arc drawing
        v1 = point1 - point2
        v2 = point3 - point2
        v1_norm = v1 / (np.linalg.norm(v1) + 1e-6)
        v2_norm = v2 / (np.linalg.norm(v2) + 1e-6)

        # Draw arc if enabled
        if self.show_arc_var.get():
            # Determine arc radius based on line lengths
            radius = min(np.linalg.norm(v1), np.linalg.norm(v2)) * 0.3
            radius = max(15, int(radius))  # Minimum radius of 15 pixels

            # Calculate start and end angles
            angle1 = np.arctan2(v1[1], v1[0])
            angle2 = np.arctan2(v2[1], v2[0])

            # Ensure proper arc direction
            if angle2 < angle1:
                angle2 += 2 * np.pi

            # Draw arc
            start_angle = angle1 * 180 / np.pi
            end_angle = angle2 * 180 / np.pi
            cv2.ellipse(frame, p2, (radius, radius), 0, start_angle, end_angle, color, 2)

        # Place angle text
        # Calculate position for text (along the bisector of the angle)
        bisector = (v1_norm + v2_norm) / 2
        if np.linalg.norm(bisector) > 0.001:  # Avoid division by zero
            bisector = bisector / np.linalg.norm(bisector)

        # Position text along bisector
        text_distance = 40 + self.angle_line_thickness * 5  # Adjust based on line thickness
        text_pos = p2 + (bisector * text_distance).astype(int)

        # Create text with background
        text = f"{angle:.1f}°"
        if label:
            text = f"{label}: {text}"

        # Draw text with contrasting background for visibility
        cv2.putText(frame, text, (text_pos[0], text_pos[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, self.angle_font_scale, (0, 0, 0),
                    self.angle_font_thickness + 2)  # Thick black outline
        cv2.putText(frame, text, (text_pos[0], text_pos[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, self.angle_font_scale, color,
                    self.angle_font_thickness)  # Color text

        return frame

    def run_analysis(self):
        if not self.video_path:
            messagebox.showerror("Error", "Please select a video file first")
            return

        # Reset results
        for key in self.results_data:
            self.results_data[key] = []

        # Update status
        self.status_var.set("Processing video...")
        self.root.update()

        # Get selected settings
        side = self.side_var.get()
        custom_point_index = self.keypoint_names.index(self.custom_point_var.get())

        # Joint indices based on side
        if side == "left":
            shoulder_idx, elbow_idx, wrist_idx = 5, 7, 9
            hip_idx, knee_idx, ankle_idx = 11, 13, 15
        else:  # right
            shoulder_idx, elbow_idx, wrist_idx = 6, 8, 10
            hip_idx, knee_idx, ankle_idx = 12, 14, 16

        try:
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                messagebox.showerror("Error", "Could not open video file")
                self.status_var.set("Ready")
                return

            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # Progress window
            progress_window = tk.Toplevel(self.root)
            progress_window.title("Processing...")
            progress_window.geometry("300x100")
            progress_window.transient(self.root)
            progress_window.resizable(False, False)

            progress_var = tk.DoubleVar()
            progress_label = ttk.Label(progress_window, text="Processing frames...")
            progress_label.pack(pady=10)
            progress_bar = ttk.Progressbar(progress_window, variable=progress_var, maximum=100)
            progress_bar.pack(fill=tk.X, padx=20)

            cancel_var = tk.BooleanVar(value=False)
            cancel_btn = ttk.Button(progress_window, text="Cancel",
                                    command=lambda: cancel_var.set(True))
            cancel_btn.pack(pady=10)

            frame_idx = 0
            processed_frames = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Skip frames for faster processing
                if frame_idx % self.skip_frames != 0:
                    frame_idx += 1
                    continue

                # Process frame
                results = self.model(frame, conf=self.confidence_threshold)[0]

                # Update progress
                progress_var.set((frame_idx / frame_count) * 100)
                progress_window.update()

                # Check for cancel
                if cancel_var.get():
                    break

                # Get a clean copy of the frame for our custom drawing
                annotated = frame.copy()

                # Check if any keypoints were detected
                if len(results.keypoints.data) > 0:
                    # Get keypoints for first person
                    kps = results.keypoints.data[0]

                    # Track custom point
                    if 0 <= custom_point_index < len(kps):
                        point = kps[custom_point_index][:2].numpy()
                        self.results_data["custom_y"].append(float(point[1]))
                        # Mark custom point
                        cv2.circle(annotated, (int(point[0]), int(point[1])), 8, (0, 0, 255), -1)
                        cv2.putText(annotated, f"{self.keypoint_names[custom_point_index]}",
                                    (int(point[0]) + 10, int(point[1])),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                    # Draw pose landmarks
                    for kp in kps:
                        x, y = int(kp[0]), int(kp[1])
                        if kp[2] > self.confidence_threshold:  # Only draw if confidence is high enough
                            cv2.circle(annotated, (x, y), 4, (0, 255, 0), -1)

                    # Connect keypoints with lines to show skeleton
                    skeleton_pairs = [
                        (5, 7), (7, 9),  # Left arm
                        (6, 8), (8, 10),  # Right arm
                        (11, 13), (13, 15),  # Left leg
                        (12, 14), (14, 16),  # Right leg
                        (5, 6), (5, 11), (6, 12), (11, 12)  # Torso
                    ]

                    for pair in skeleton_pairs:
                        if all(0 <= i < len(kps) and kps[i][2] > self.confidence_threshold for i in pair):
                            pt1 = (int(kps[pair[0]][0]), int(kps[pair[0]][1]))
                            pt2 = (int(kps[pair[1]][0]), int(kps[pair[1]][1]))
                            cv2.line(annotated, pt1, pt2, (0, 255, 0), 2)

                    # Calculate elbow angle
                    if self.track_elbow.get() and all(0 <= i < len(kps) for i in [shoulder_idx, elbow_idx, wrist_idx]):
                        shoulder = kps[shoulder_idx][:2].numpy()
                        elbow = kps[elbow_idx][:2].numpy()
                        wrist = kps[wrist_idx][:2].numpy()

                        elbow_angle = self.calculate_angle(shoulder, elbow, wrist)
                        self.results_data["elbow_angles"].append(elbow_angle)

                        # Draw enhanced angle visualization
                        annotated = self.draw_angle(
                            annotated, shoulder, elbow, wrist,
                            elbow_angle, self.angle_colors["elbow"], "Elbow"
                        )

                    # Calculate knee angle
                    if self.track_knee.get() and all(0 <= i < len(kps) for i in [hip_idx, knee_idx, ankle_idx]):
                        hip = kps[hip_idx][:2].numpy()
                        knee = kps[knee_idx][:2].numpy()
                        ankle = kps[ankle_idx][:2].numpy()

                        knee_angle = self.calculate_angle(hip, knee, ankle)
                        self.results_data["knee_angles"].append(knee_angle)

                        # Draw enhanced angle visualization
                        annotated = self.draw_angle(
                            annotated, hip, knee, ankle,
                            knee_angle, self.angle_colors["knee"], "Knee"
                        )

                    # Calculate hip angle (between shoulder, hip, and knee)
                    if self.track_hip.get() and all(0 <= i < len(kps) for i in [shoulder_idx, hip_idx, knee_idx]):
                        shoulder = kps[shoulder_idx][:2].numpy()
                        hip = kps[hip_idx][:2].numpy()
                        knee = kps[knee_idx][:2].numpy()

                        hip_angle = self.calculate_angle(shoulder, hip, knee)
                        self.results_data["hip_angles"].append(hip_angle)

                        # Draw enhanced angle visualization
                        annotated = self.draw_angle(
                            annotated, shoulder, hip, knee,
                            hip_angle, self.angle_colors["hip"], "Hip"
                        )

                # Add timestamp
                self.results_data["timestamps"].append(frame_idx / fps)

                # Add timestamp to frame
                cv2.putText(annotated, f"Time: {frame_idx / fps:.2f}s", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                # Resize for display if needed
                if self.video_scale != 1.0:
                    h, w = annotated.shape[:2]
                    new_w, new_h = int(w * self.video_scale), int(h * self.video_scale)
                    display_frame = cv2.resize(annotated, (new_w, new_h))
                else:
                    display_frame = annotated

                # Show frame
                cv2.imshow("Pose Analysis", display_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                frame_idx += 1
                processed_frames += 1

            # Clean up
            cap.release()
            cv2.destroyAllWindows()
            progress_window.destroy()

            # Show results if not canceled
            if not cancel_var.get() and processed_frames > 0:
                self.show_results()
                self.status_var.set(f"Analysis complete: {processed_frames} frames processed")
            else:
                self.status_var.set("Analysis canceled")

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred during analysis: {str(e)}")
            self.status_var.set("Analysis failed")

    def show_results(self):
        # Create results window
        results_win = tk.Toplevel(self.root)
        results_win.title("Analysis Results")
        results_win.geometry("800x600")

        # Create notebook for tabs
        tab_control = ttk.Notebook(results_win)

        # Create tabs
        graph_tab = ttk.Frame(tab_control)
        data_tab = ttk.Frame(tab_control)

        tab_control.add(graph_tab, text="Graphs")
        tab_control.add(data_tab, text="Data Table")
        tab_control.pack(expand=1, fill=tk.BOTH)

        # Create graphs
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot data if available
        plots = []
        labels = []

        if self.results_data["elbow_angles"] and self.track_elbow.get():
            time_data = self.results_data["timestamps"][:len(self.results_data["elbow_angles"])]
            plot_elbow, = ax.plot(time_data, self.results_data["elbow_angles"], color='blue', label="Elbow Angle")
            plots.append(plot_elbow)
            labels.append("Elbow Angle")

        if self.results_data["knee_angles"] and self.track_knee.get():
            time_data = self.results_data["timestamps"][:len(self.results_data["knee_angles"])]
            plot_knee, = ax.plot(time_data, self.results_data["knee_angles"], color='red', label="Knee Angle")
            plots.append(plot_knee)
            labels.append("Knee Angle")

        if self.results_data["hip_angles"] and self.track_hip.get():
            time_data = self.results_data["timestamps"][:len(self.results_data["hip_angles"])]
            plot_hip, = ax.plot(time_data, self.results_data["hip_angles"], color='green', label="Hip Angle")
            plots.append(plot_hip)
            labels.append("Hip Angle")

        if self.results_data["custom_y"]:
            time_data = self.results_data["timestamps"][:len(self.results_data["custom_y"])]
            # Normalize custom point data for visual comparison
            custom_data = np.array(self.results_data["custom_y"])
            if len(custom_data) > 0:
                custom_data = (custom_data - np.min(custom_data)) / (
                            np.max(custom_data) - np.min(custom_data) + 1e-6) * 100
                plot_custom, = ax.plot(time_data, custom_data, color='purple', linestyle='--',
                                       label=f"{self.custom_point_var.get()} Y-position (normalized)")
                plots.append(plot_custom)
                labels.append(f"{self.custom_point_var.get()} Y-position")

        ax.set_title("Joint Angles and Position Over Time")
        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Angle (degrees) / Position (normalized)")
        ax.grid(True)

        if plots:
            ax.legend()
        else:
            ax.text(0.5, 0.5, "No data available", ha='center', va='center', transform=ax.transAxes)

        # Add graph to tab
        canvas = FigureCanvasTkAgg(fig, master=graph_tab)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Create toolbar frame
        toolbar_frame = ttk.Frame(graph_tab)
        toolbar_frame.pack(fill=tk.X, padx=10)

        # Add checkboxes to toggle visibility
        checkbox_frame = ttk.LabelFrame(toolbar_frame, text="Toggle Series")
        checkbox_frame.pack(side=tk.LEFT, padx=10, pady=5)

        # Create visibility toggles for each plot
        for i, (plot, label) in enumerate(zip(plots, labels)):
            var = tk.BooleanVar(value=True)
            cb = ttk.Checkbutton(checkbox_frame, text=label, variable=var,
                                 command=lambda p=plot, v=var: self.toggle_plot_visibility(p, v.get()))
            cb.pack(side=tk.LEFT, padx=10)

        # Create data table
        tree_frame = ttk.Frame(data_tab)
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Create treeview
        columns = ["Time (s)"]
        if self.track_elbow.get():
            columns.append("Elbow Angle (°)")
        if self.track_knee.get():
            columns.append("Knee Angle (°)")
        if self.track_hip.get():
            columns.append("Hip Angle (°)")
        columns.append(f"{self.custom_point_var.get()} Y")

        tree = ttk.Treeview(tree_frame, columns=columns, show="headings")

        # Set column headings
        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=100)

        # Add data to treeview
        max_len = max(len(self.results_data["timestamps"]),
                      len(self.results_data["elbow_angles"]),
                      len(self.results_data["knee_angles"]),
                      len(self.results_data["hip_angles"]),
                      len(self.results_data["custom_y"]))

        for i in range(min(max_len, 1000)):  # Limit to 1000 rows for performance
            values = []

            # Time
            if i < len(self.results_data["timestamps"]):
                values.append(f"{self.results_data['timestamps'][i]:.2f}")
            else:
                values.append("")

            # Elbow
            if self.track_elbow.get():
                if i < len(self.results_data["elbow_angles"]):
                    values.append(f"{self.results_data['elbow_angles'][i]:.2f}")
                else:
                    values.append("")

            # Knee
            if self.track_knee.get():
                if i < len(self.results_data["knee_angles"]):
                    values.append(f"{self.results_data['knee_angles'][i]:.2f}")
                else:
                    values.append("")

            # Hip
            if self.track_hip.get():
                if i < len(self.results_data["hip_angles"]):
                    values.append(f"{self.results_data['hip_angles'][i]:.2f}")
                else:
                    values.append("")

            # Custom Y
            if i < len(self.results_data["custom_y"]):
                values.append(f"{self.results_data['custom_y'][i]:.2f}")
            else:
                values.append("")

            tree.insert("", tk.END, values=values)

        # Add scrollbar
        scrollbar = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        tree.pack(fill=tk.BOTH, expand=True)

        # Show note if data was truncated
        if max_len > 1000:
            ttk.Label(data_tab, text=f"Note: Only showing first 1000 of {max_len} data points").pack(pady=5)

    def toggle_plot_visibility(self, plot, visible):
        plot.set_visible(visible)
        plot.figure.canvas.draw()

    def export_data(self):
        if not any(len(self.results_data[key]) > 0 for key in self.results_data if key != "timestamps"):
            messagebox.showinfo("Export", "No data available to export")
            return

        try:
            # Create filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            default_filename = f"pose_analysis_{timestamp}.csv"

            # Get save path
            file_path = filedialog.asksaveasfilename(
                initialdir=self.export_path,
                initialfile=default_filename,
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
            )

            if not file_path:
                return

            # Prepare data for export
            export_data = []
            headers = ["Time (s)"]

            # Add column headers
            if self.results_data["elbow_angles"]:
                headers.append("Elbow Angle (degrees)")
            if self.results_data["knee_angles"]:
                headers.append("Knee Angle (degrees)")
            if self.results_data["hip_angles"]:
                headers.append("Hip Angle (degrees)")
            if self.results_data["custom_y"]:
                headers.append(f"{self.custom_point_var.get()} Y Position")

            # Get max length of data
            max_len = max(len(self.results_data[key]) for key in self.results_data)

            # Create rows
            for i in range(max_len):
                row = []

                # Time
                if i < len(self.results_data["timestamps"]):
                    row.append(self.results_data["timestamps"][i])
                else:
                    row.append("")

                # Elbow angles
                if self.results_data["elbow_angles"]:
                    if i < len(self.results_data["elbow_angles"]):
                        row.append(self.results_data["elbow_angles"][i])
                    else:
                        row.append("")

                # Knee angles
                if self.results_data["knee_angles"]:
                    if i < len(self.results_data["knee_angles"]):
                        row.append(self.results_data["knee_angles"][i])
                    else:
                        row.append("")

                # Hip angles
                if self.results_data["hip_angles"]:
                    if i < len(self.results_data["hip_angles"]):
                        row.append(self.results_data["hip_angles"][i])
                    else:
                        row.append("")

                # Custom Y position
                if self.results_data["custom_y"]:
                    if i < len(self.results_data["custom_y"]):
                        row.append(self.results_data["custom_y"][i])
                    else:
                        row.append("")

                export_data.append(row)

            # Write to CSV
            with open(file_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(headers)
                writer.writerows(export_data)

            messagebox.showinfo("Export Successful", f"Data exported to {file_path}")

        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export data: {str(e)}")


if __name__ == "__main__":
    root = tk.Tk()
    app = PoseAnalyzer(root)
    root.mainloop()
