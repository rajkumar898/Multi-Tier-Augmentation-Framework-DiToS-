"""
================================================================================
DiToS MULTI-CLASS DATASET VISUALIZATION - DETAILED FLOW
================================================================================
Shows:
1. Original dataset distribution (792 images, 6 classes)
2. Selected images for each imbalance case
3. DiToS pipeline stages: Original → SD (+40%) → Tomek → SMOTE
4. Final balanced distribution

Publication-ready figures at 300 DPI
================================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as path_effects
import seaborn as sns
import os

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 10,
    'font.family': 'serif',
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

OUTPUT_DIR = "./DiToS_Visualizations_Detailed"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==============================================================================
# DATASET CONFIGURATION - UPDATE WITH YOUR ACTUAL NUMBERS
# ==============================================================================

# Original pythonafroz dataset
ORIGINAL_DATASET = {
    'Clean': 200,
    'Dusty': 180,
    'Physical-Damage': 130,
    'Electrical-Damage': 120,
    'Bird-drop': 90,
    'Snow-Covered': 72
}

TOTAL_ORIGINAL = sum(ORIGINAL_DATASET.values())  # 792

# Colors for each class
CLASS_COLORS = {
    'Clean': '#27ae60',           # Green
    'Dusty': '#c0392b',           # Dark Red
    'Physical-Damage': '#2980b9', # Blue
    'Electrical-Damage': '#f39c12', # Orange
    'Bird-drop': '#8e44ad',       # Purple
    'Snow-Covered': '#16a085'     # Teal
}

# Stage colors
STAGE_COLORS = {
    'original': '#e74c3c',      # Red
    'sd': '#f39c12',            # Orange
    'tomek': '#3498db',         # Blue
    'smote': '#2ecc71'          # Green
}

# DiToS parameters
SD_RATIO = 0.40  # 40% from Stable Diffusion


# ==============================================================================
# CALCULATE DISTRIBUTIONS FOR EACH CASE
# ==============================================================================

def calculate_case_distributions():
    """Calculate distributions for all three cases."""
    
    cases = {
        'Case 1 (IR=10:1)': {'ir': 10.0, 'data': {}},
        'Case 2 (IR=3:1)': {'ir': 3.0, 'data': {}},
        'Case 3 (IR=1.9:1)': {'ir': 1.9, 'data': {}}
    }
    
    majority_class = 'Clean'
    majority_count = ORIGINAL_DATASET[majority_class]
    
    for case_name, case_info in cases.items():
        target_ir = case_info['ir']
        target_minority = int(majority_count / target_ir)
        
        for cls, orig_count in ORIGINAL_DATASET.items():
            if cls == majority_class:
                selected = orig_count
            else:
                selected = min(orig_count, target_minority)
            
            # Calculate DiToS stages
            deficit = majority_count - selected
            sd_added = int(deficit * SD_RATIO) if deficit > 0 else 0
            after_sd = selected + sd_added
            
            # Tomek removes ~3-5% boundary samples
            tomek_removed = int(after_sd * 0.03) if cls != majority_class else int(after_sd * 0.05)
            after_tomek = after_sd - tomek_removed
            
            # SMOTE balances to majority class level (after tomek)
            majority_after_tomek = majority_count - int(majority_count * 0.05)
            smote_added = max(0, majority_after_tomek - after_tomek)
            after_smote = after_tomek + smote_added
            
            case_info['data'][cls] = {
                'original': orig_count,
                'selected': selected,
                'sd_added': sd_added,
                'after_sd': after_sd,
                'tomek_removed': tomek_removed,
                'after_tomek': after_tomek,
                'smote_added': smote_added,
                'after_smote': after_smote
            }
    
    return cases


# ==============================================================================
# FIGURE 1: ORIGINAL DATASET OVERVIEW
# ==============================================================================

def plot_original_dataset():
    """Plot original dataset distribution with pie and bar."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    classes = list(ORIGINAL_DATASET.keys())
    counts = list(ORIGINAL_DATASET.values())
    colors = [CLASS_COLORS[c] for c in classes]
    
    # Pie chart
    ax1 = axes[0]
    wedges, texts, autotexts = ax1.pie(
        counts, labels=classes, colors=colors, autopct='%1.1f%%',
        startangle=90, explode=[0.02]*6,
        textprops={'fontsize': 9}
    )
    ax1.set_title(f'Original Dataset Distribution\n(Total: {TOTAL_ORIGINAL} images)', 
                  fontsize=13, fontweight='bold')
    
    # Bar chart
    ax2 = axes[1]
    bars = ax2.bar(classes, counts, color=colors, edgecolor='black', linewidth=1.2)
    
    # Add count and percentage labels
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        pct = count / TOTAL_ORIGINAL * 100
        ax2.annotate(f'{count}\n({pct:.1f}%)',
                     xy=(bar.get_x() + bar.get_width()/2, height),
                     xytext=(0, 5), textcoords="offset points",
                     ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax2.set_ylabel('Number of Images', fontsize=11, fontweight='bold')
    ax2.set_xlabel('Fault Class', fontsize=11, fontweight='bold')
    ax2.set_title('Class-wise Image Count', fontsize=13, fontweight='bold')
    ax2.set_ylim(0, max(counts) * 1.25)
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=20, ha='right')
    
    plt.suptitle('Multi-Class Solar Panel Fault Detection Dataset (pythonafroz)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    plt.savefig(os.path.join(OUTPUT_DIR, '01_original_dataset.png'), dpi=300)
    plt.savefig(os.path.join(OUTPUT_DIR, '01_original_dataset.pdf'))
    plt.close()
    print("✓ Saved: 01_original_dataset.png")


# ==============================================================================
# FIGURE 2: IMBALANCE CASE SELECTION
# ==============================================================================

def plot_imbalance_selection():
    """Show how images are selected for each imbalance case."""
    cases = calculate_case_distributions()
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    
    classes = list(ORIGINAL_DATASET.keys())
    
    for idx, (case_name, case_info) in enumerate(cases.items()):
        ax = axes[idx]
        
        selected = [case_info['data'][c]['selected'] for c in classes]
        colors = [CLASS_COLORS[c] for c in classes]
        
        bars = ax.bar(classes, selected, color=colors, edgecolor='black', linewidth=1)
        
        # Add labels
        for bar, count in zip(bars, selected):
            ax.annotate(f'{count}',
                        xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', fontsize=10, fontweight='bold')
        
        total = sum(selected)
        ax.set_title(f'{case_name}\nTotal: {total} images', fontsize=12, fontweight='bold')
        ax.set_xlabel('Fault Class', fontsize=10)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=25, ha='right')
        
        if idx == 0:
            ax.set_ylabel('Number of Images', fontsize=11, fontweight='bold')
    
    plt.suptitle('Images Selected for Each Imbalance Scenario\n(Majority: Clean = 200, Minority classes reduced)',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    plt.savefig(os.path.join(OUTPUT_DIR, '02_imbalance_selection.png'), dpi=300)
    plt.savefig(os.path.join(OUTPUT_DIR, '02_imbalance_selection.pdf'))
    plt.close()
    print("✓ Saved: 02_imbalance_selection.png")


# ==============================================================================
# FIGURE 3: DiToS PIPELINE FLOW FOR EACH CASE
# ==============================================================================

def plot_ditos_pipeline_detailed(case_name, case_info, filename):
    """Create detailed DiToS pipeline visualization for one case."""
    
    fig, ax = plt.subplots(figsize=(16, 9))
    
    classes = list(ORIGINAL_DATASET.keys())
    n_classes = len(classes)
    
    # Stage positions (x-axis)
    stages = ['Selected\n(Imbalanced)', '+40% SD\nImages', 'After Tomek\n(Cleaned)', 'After SMOTE\n(Balanced)']
    stage_keys = ['selected', 'after_sd', 'after_tomek', 'after_smote']
    n_stages = len(stages)
    
    x = np.arange(n_stages)
    width = 0.12
    
    # Plot bars for each class
    for i, cls in enumerate(classes):
        data = case_info['data'][cls]
        counts = [data[key] for key in stage_keys]
        
        offset = (i - n_classes/2 + 0.5) * width
        bars = ax.bar(x + offset, counts, width, label=cls, 
                      color=CLASS_COLORS[cls], edgecolor='black', linewidth=0.8)
        
        # Add count labels
        for bar, count in zip(bars, counts):
            if count > 0:
                ax.annotate(f'{count}',
                            xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                            xytext=(0, 2), textcoords="offset points",
                            ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    # Add stage totals
    for stage_idx, stage_key in enumerate(stage_keys):
        total = sum(case_info['data'][c][stage_key] for c in classes)
        ax.annotate(f'Total: {total}',
                    xy=(stage_idx, ax.get_ylim()[1] * 0.95),
                    ha='center', fontsize=11, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Add arrows showing changes
    arrow_y = ax.get_ylim()[1] * 0.85
    for i in range(n_stages - 1):
        ax.annotate('', xy=(i + 0.7, arrow_y), xytext=(i + 0.3, arrow_y),
                    arrowprops=dict(arrowstyle='->', color='black', lw=2))
    
    ax.set_xticks(x)
    ax.set_xticklabels(stages, fontsize=11, fontweight='bold')
    ax.set_ylabel('Number of Images', fontsize=12, fontweight='bold')
    ax.set_title(f'DiToS Pipeline Stages - {case_name}\n'
                 f'Strategy: 40% Stable Diffusion + Tomek Links + SMOTE',
                 fontsize=14, fontweight='bold')
    
    ax.legend(loc='upper left', ncol=3, fontsize=9, framealpha=0.9)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.15)
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    plt.savefig(os.path.join(OUTPUT_DIR, filename + '.png'), dpi=300)
    plt.savefig(os.path.join(OUTPUT_DIR, filename + '.pdf'))
    plt.close()
    print(f"✓ Saved: {filename}.png")


# ==============================================================================
# FIGURE 4: CONTRIBUTION BREAKDOWN
# ==============================================================================

def plot_contribution_breakdown():
    """Show contribution of each DiToS component per case."""
    cases = calculate_case_distributions()
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    classes = list(ORIGINAL_DATASET.keys())
    
    for idx, (case_name, case_info) in enumerate(cases.items()):
        ax = axes[idx]
        
        # Get data
        original = [case_info['data'][c]['selected'] for c in classes]
        sd_added = [case_info['data'][c]['sd_added'] for c in classes]
        smote_added = [case_info['data'][c]['smote_added'] for c in classes]
        
        x = np.arange(len(classes))
        
        # Stacked bar
        bars1 = ax.bar(x, original, label='Original', color='#3498db', 
                       edgecolor='black', linewidth=0.8)
        bars2 = ax.bar(x, sd_added, bottom=original, label='SD Added (+40%)', 
                       color='#e74c3c', edgecolor='black', linewidth=0.8)
        
        # SMOTE on top (after tomek adjustment)
        bottom_smote = [case_info['data'][c]['after_tomek'] for c in classes]
        bars3 = ax.bar(x, smote_added, bottom=bottom_smote, label='SMOTE Added',
                       color='#2ecc71', edgecolor='black', linewidth=0.8)
        
        # Final count labels
        for i, c in enumerate(classes):
            final = case_info['data'][c]['after_smote']
            ax.annotate(f'{final}',
                        xy=(i, final + 3),
                        ha='center', fontsize=9, fontweight='bold')
        
        ax.set_title(case_name, fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(classes, rotation=30, ha='right', fontsize=9)
        ax.set_ylabel('Number of Images' if idx == 0 else '', fontsize=11)
        
        if idx == 2:
            ax.legend(loc='upper right', fontsize=9)
    
    plt.suptitle('DiToS Augmentation: Contribution by Component\n'
                 '(Original + 40% Stable Diffusion + SMOTE)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    plt.savefig(os.path.join(OUTPUT_DIR, '04_contribution_breakdown.png'), dpi=300)
    plt.savefig(os.path.join(OUTPUT_DIR, '04_contribution_breakdown.pdf'))
    plt.close()
    print("✓ Saved: 04_contribution_breakdown.png")


# ==============================================================================
# FIGURE 5: COMPREHENSIVE FLOW DIAGRAM
# ==============================================================================

def plot_comprehensive_flow():
    """Create comprehensive flow diagram showing entire process."""
    cases = calculate_case_distributions()
    
    fig = plt.figure(figsize=(20, 14))
    
    # Create grid
    gs = GridSpec(3, 4, figure=fig, hspace=0.35, wspace=0.25,
                  height_ratios=[1, 1, 1.2])
    
    classes = list(ORIGINAL_DATASET.keys())
    colors = [CLASS_COLORS[c] for c in classes]
    
    # Row 0: Original Dataset
    ax_orig = fig.add_subplot(gs[0, :2])
    counts = list(ORIGINAL_DATASET.values())
    bars = ax_orig.bar(classes, counts, color=colors, edgecolor='black', linewidth=1)
    for bar, count in zip(bars, counts):
        ax_orig.annotate(f'{count}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                         xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9, fontweight='bold')
    ax_orig.set_title(f'(A) Original Dataset: {TOTAL_ORIGINAL} images', fontsize=12, fontweight='bold')
    ax_orig.set_ylabel('Images', fontsize=10)
    plt.setp(ax_orig.xaxis.get_majorticklabels(), rotation=20, ha='right')
    
    # Row 0: Imbalance explanation
    ax_text = fig.add_subplot(gs[0, 2:])
    ax_text.axis('off')
    
    text = """
    IMBALANCE SCENARIOS:
    
    Case 1 (Severe):    IR = 10:1
        Majority (Clean) = 200
        Each Minority    = 20
        Total: 300 images
    
    Case 2 (Moderate):  IR = 3:1
        Majority (Clean) = 200
        Each Minority    = 66
        Total: 530 images
    
    Case 3 (Mild):      IR = 1.9:1
        Majority (Clean) = 200
        Minorities       = 72-105
        Total: 677 images
    """
    ax_text.text(0.1, 0.5, text, fontsize=11, family='monospace',
                 verticalalignment='center',
                 bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    ax_text.set_title('(B) Imbalance Case Configuration', fontsize=12, fontweight='bold')
    
    # Row 1: Three cases - selected images
    for idx, (case_name, case_info) in enumerate(cases.items()):
        ax = fig.add_subplot(gs[1, idx])
        
        selected = [case_info['data'][c]['selected'] for c in classes]
        bars = ax.bar(classes, selected, color=colors, edgecolor='black', linewidth=0.8)
        
        for bar, count in zip(bars, selected):
            ax.annotate(f'{count}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                        xytext=(0, 2), textcoords="offset points", ha='center', fontsize=8, fontweight='bold')
        
        total = sum(selected)
        short_name = case_name.split('(')[0].strip()
        ax.set_title(f'(C{idx+1}) {short_name}\nTotal: {total}', fontsize=11, fontweight='bold')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right', fontsize=8)
        
        if idx == 0:
            ax.set_ylabel('Images', fontsize=10)
    
    # Row 1: Arrow to DiToS
    ax_arrow = fig.add_subplot(gs[1, 3])
    ax_arrow.axis('off')
    ax_arrow.annotate('', xy=(0.1, 0.5), xytext=(0.9, 0.5),
                      arrowprops=dict(arrowstyle='->', color='red', lw=4))
    ax_arrow.text(0.5, 0.7, 'Apply\nDiToS', fontsize=14, ha='center', fontweight='bold', color='red')
    ax_arrow.text(0.5, 0.3, '40% SD\n+ Tomek\n+ SMOTE', fontsize=10, ha='center')
    
    # Row 2: Final balanced results
    ax_final = fig.add_subplot(gs[2, :])
    
    case_labels = ['Case 1\n(10:1)', 'Case 2\n(3:1)', 'Case 3\n(1.9:1)']
    x = np.arange(3)
    width = 0.12
    
    for i, cls in enumerate(classes):
        offset = (i - len(classes)/2 + 0.5) * width
        final_counts = []
        for case_name, case_info in cases.items():
            final_counts.append(case_info['data'][cls]['after_smote'])
        
        bars = ax_final.bar(x + offset, final_counts, width, label=cls,
                            color=CLASS_COLORS[cls], edgecolor='black', linewidth=0.8)
    
    # Add total labels
    for case_idx, (case_name, case_info) in enumerate(cases.items()):
        total = sum(case_info['data'][c]['after_smote'] for c in classes)
        ax_final.annotate(f'Total: {total}',
                          xy=(case_idx, ax_final.get_ylim()[1] if ax_final.get_ylim()[1] > 0 else 200),
                          ha='center', fontsize=11, fontweight='bold',
                          bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    ax_final.set_xticks(x)
    ax_final.set_xticklabels(case_labels, fontsize=11)
    ax_final.set_ylabel('Images per Class', fontsize=11, fontweight='bold')
    ax_final.set_title('(D) Final Balanced Dataset After DiToS\n(All classes balanced to ~190 images)',
                       fontsize=12, fontweight='bold')
    ax_final.legend(loc='upper right', ncol=6, fontsize=9)
    ax_final.set_ylim(0, 220)
    
    plt.suptitle('DiToS Multi-Class Dataset Flow: From Original to Balanced\n'
                 'Diffusion-Tomek-SMOTE Framework',
                 fontsize=15, fontweight='bold', y=0.98)
    
    plt.savefig(os.path.join(OUTPUT_DIR, '05_comprehensive_flow.png'), dpi=300)
    plt.savefig(os.path.join(OUTPUT_DIR, '05_comprehensive_flow.pdf'))
    plt.close()
    print("✓ Saved: 05_comprehensive_flow.png")


# ==============================================================================
# FIGURE 6: DETAILED NUMBERS TABLE
# ==============================================================================

def plot_detailed_table():
    """Create detailed table with all numbers."""
    cases = calculate_case_distributions()
    
    fig, ax = plt.subplots(figsize=(18, 12))
    ax.axis('off')
    
    classes = list(ORIGINAL_DATASET.keys())
    
    # Prepare table data
    columns = ['Case', 'Class', 'Original', 'Selected', 'SD Added', 'After SD', 
               'Tomek Removed', 'After Tomek', 'SMOTE Added', 'Final']
    
    data = []
    for case_name, case_info in cases.items():
        short_name = case_name.split('(')[0].strip()
        for cls in classes:
            d = case_info['data'][cls]
            data.append([
                short_name,
                cls,
                d['original'],
                d['selected'],
                f"+{d['sd_added']}" if d['sd_added'] > 0 else '0',
                d['after_sd'],
                f"-{d['tomek_removed']}" if d['tomek_removed'] > 0 else '0',
                d['after_tomek'],
                f"+{d['smote_added']}" if d['smote_added'] > 0 else '0',
                d['after_smote']
            ])
        
        # Add total row
        total_selected = sum(case_info['data'][c]['selected'] for c in classes)
        total_final = sum(case_info['data'][c]['after_smote'] for c in classes)
        data.append([short_name, 'TOTAL', '-', total_selected, '-', '-', '-', '-', '-', total_final])
    
    # Create table
    table = ax.table(
        cellText=data,
        colLabels=columns,
        cellLoc='center',
        loc='center',
        colColours=['#3498db'] * len(columns)
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.8)
    
    # Style header
    for i in range(len(columns)):
        table[(0, i)].set_text_props(color='white', fontweight='bold')
    
    # Alternate row colors and highlight totals
    for i in range(1, len(data) + 1):
        for j in range(len(columns)):
            if data[i-1][1] == 'TOTAL':
                table[(i, j)].set_facecolor('#ffffcc')
                table[(i, j)].set_text_props(fontweight='bold')
            elif i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
    
    ax.set_title('DiToS Multi-Class Augmentation: Complete Sample Counts\n'
                 'Original → Selected → +40% SD → Tomek Links → SMOTE → Final Balanced',
                 fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    plt.savefig(os.path.join(OUTPUT_DIR, '06_detailed_table.png'), dpi=300)
    plt.savefig(os.path.join(OUTPUT_DIR, '06_detailed_table.pdf'))
    plt.close()
    
    # Also save as CSV
    df = pd.DataFrame(data, columns=columns)
    df.to_csv(os.path.join(OUTPUT_DIR, 'ditos_complete_counts.csv'), index=False)
    
    print("✓ Saved: 06_detailed_table.png")
    print("✓ Saved: ditos_complete_counts.csv")


# ==============================================================================
# FIGURE 7: BEFORE vs AFTER COMPARISON
# ==============================================================================

def plot_before_after_comparison():
    """Direct comparison of before and after DiToS."""
    cases = calculate_case_distributions()
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    classes = list(ORIGINAL_DATASET.keys())
    colors = [CLASS_COLORS[c] for c in classes]
    
    for idx, (case_name, case_info) in enumerate(cases.items()):
        # Before DiToS
        ax_before = axes[0, idx]
        selected = [case_info['data'][c]['selected'] for c in classes]
        bars = ax_before.bar(classes, selected, color=colors, edgecolor='black', linewidth=1)
        
        for bar, count in zip(bars, selected):
            ax_before.annotate(f'{count}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                               xytext=(0, 2), textcoords="offset points", ha='center', fontsize=9, fontweight='bold')
        
        total_before = sum(selected)
        short_name = case_name.split('(')[0].strip()
        ax_before.set_title(f'{short_name} - BEFORE DiToS\nTotal: {total_before}', 
                            fontsize=11, fontweight='bold', color='red')
        ax_before.set_ylim(0, 220)
        plt.setp(ax_before.xaxis.get_majorticklabels(), rotation=25, ha='right', fontsize=8)
        
        if idx == 0:
            ax_before.set_ylabel('Number of Images', fontsize=10, fontweight='bold')
        
        # After DiToS
        ax_after = axes[1, idx]
        final = [case_info['data'][c]['after_smote'] for c in classes]
        bars = ax_after.bar(classes, final, color=colors, edgecolor='black', linewidth=1)
        
        for bar, count in zip(bars, final):
            ax_after.annotate(f'{count}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                              xytext=(0, 2), textcoords="offset points", ha='center', fontsize=9, fontweight='bold')
        
        total_after = sum(final)
        ax_after.set_title(f'{short_name} - AFTER DiToS\nTotal: {total_after}', 
                           fontsize=11, fontweight='bold', color='green')
        ax_after.set_ylim(0, 220)
        plt.setp(ax_after.xaxis.get_majorticklabels(), rotation=25, ha='right', fontsize=8)
        
        if idx == 0:
            ax_after.set_ylabel('Number of Images', fontsize=10, fontweight='bold')
    
    plt.suptitle('Before vs After DiToS Augmentation\n'
                 '(40% Stable Diffusion + Tomek Links + SMOTE)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    plt.savefig(os.path.join(OUTPUT_DIR, '07_before_after_comparison.png'), dpi=300)
    plt.savefig(os.path.join(OUTPUT_DIR, '07_before_after_comparison.pdf'))
    plt.close()
    print("✓ Saved: 07_before_after_comparison.png")


# ==============================================================================
# FIGURE 8: SANKEY-STYLE FLOW
# ==============================================================================

def plot_augmentation_flow_sankey():
    """Create Sankey-style flow showing augmentation contributions."""
    cases = calculate_case_distributions()
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 8))
    
    classes = list(ORIGINAL_DATASET.keys())
    
    for idx, (case_name, case_info) in enumerate(cases.items()):
        ax = axes[idx]
        
        # Data for waterfall
        categories = ['Selected', '+SD (40%)', '-Tomek', '+SMOTE', 'Final']
        
        # Calculate totals at each stage
        total_selected = sum(case_info['data'][c]['selected'] for c in classes)
        total_sd = sum(case_info['data'][c]['sd_added'] for c in classes)
        total_tomek = sum(case_info['data'][c]['tomek_removed'] for c in classes)
        total_smote = sum(case_info['data'][c]['smote_added'] for c in classes)
        total_final = sum(case_info['data'][c]['after_smote'] for c in classes)
        
        values = [total_selected, total_sd, -total_tomek, total_smote, total_final]
        
        # Running total for waterfall
        running = [total_selected]
        running.append(running[-1] + total_sd)
        running.append(running[-1] - total_tomek)
        running.append(running[-1] + total_smote)
        
        colors_bar = ['#3498db', '#2ecc71', '#e74c3c', '#2ecc71', '#27ae60']
        
        # Create waterfall-style bars
        bottoms = [0, total_selected, total_selected + total_sd, 
                   total_selected + total_sd - total_tomek, 0]
        heights = [total_selected, total_sd, total_tomek, total_smote, total_final]
        
        x = np.arange(len(categories))
        
        for i, (cat, h, b, c) in enumerate(zip(categories, heights, bottoms, colors_bar)):
            if i == 4:  # Final bar from 0
                ax.bar(i, h, color=c, edgecolor='black', linewidth=1.5)
            elif i == 2:  # Tomek removal (negative)
                ax.bar(i, h, bottom=b-h, color=c, edgecolor='black', linewidth=1.5)
            else:
                ax.bar(i, h, bottom=b, color=c, edgecolor='black', linewidth=1.5)
        
        # Labels
        labels = [f'{total_selected}', f'+{total_sd}', f'-{total_tomek}', 
                  f'+{total_smote}', f'{total_final}']
        positions = [total_selected/2, total_selected + total_sd/2,
                     total_selected + total_sd - total_tomek/2,
                     total_selected + total_sd - total_tomek + total_smote/2,
                     total_final/2]
        
        for i, (label, pos) in enumerate(zip(labels, positions)):
            ax.annotate(label, xy=(i, pos), ha='center', va='center',
                        fontsize=11, fontweight='bold', color='white',
                        bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))
        
        short_name = case_name.split('(')[0].strip()
        ax.set_title(f'{case_name}\n{total_selected} → {total_final}',
                     fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(categories, fontsize=10, fontweight='bold')
        ax.set_ylabel('Total Images' if idx == 0 else '', fontsize=11)
        ax.set_ylim(0, max(total_final, total_selected + total_sd) * 1.1)
    
    plt.suptitle('DiToS Augmentation Flow: Total Images at Each Stage',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    plt.savefig(os.path.join(OUTPUT_DIR, '08_augmentation_flow.png'), dpi=300)
    plt.savefig(os.path.join(OUTPUT_DIR, '08_augmentation_flow.pdf'))
    plt.close()
    print("✓ Saved: 08_augmentation_flow.png")


# ==============================================================================
# PRINT SUMMARY
# ==============================================================================

def print_summary():
    """Print detailed summary to console."""
    cases = calculate_case_distributions()
    classes = list(ORIGINAL_DATASET.keys())
    
    print("\n" + "="*100)
    print("DiToS MULTI-CLASS DATASET SUMMARY")
    print("="*100)
    
    print(f"\nORIGINAL DATASET: {TOTAL_ORIGINAL} images")
    for cls, count in ORIGINAL_DATASET.items():
        print(f"  {cls:<20}: {count:>4} ({count/TOTAL_ORIGINAL*100:.1f}%)")
    
    print("\n" + "-"*100)
    
    header = f"{'Case':<15} {'Class':<20} {'Orig':>6} {'Select':>6} {'SD+':>6} {'After SD':>8} {'Tomek-':>7} {'After T':>8} {'SMOTE+':>7} {'Final':>6}"
    print(header)
    print("-"*100)
    
    for case_name, case_info in cases.items():
        short_name = case_name.split('(')[0].strip()
        for cls in classes:
            d = case_info['data'][cls]
            print(f"{short_name:<15} {cls:<20} {d['original']:>6} {d['selected']:>6} "
                  f"{'+'+str(d['sd_added']):>6} {d['after_sd']:>8} "
                  f"{'-'+str(d['tomek_removed']):>7} {d['after_tomek']:>8} "
                  f"{'+'+str(d['smote_added']):>7} {d['after_smote']:>6}")
        
        # Total row
        total_sel = sum(case_info['data'][c]['selected'] for c in classes)
        total_final = sum(case_info['data'][c]['after_smote'] for c in classes)
        print(f"{'':<15} {'TOTAL':<20} {'-':>6} {total_sel:>6} {'-':>6} {'-':>8} {'-':>7} {'-':>8} {'-':>7} {total_final:>6}")
        print("-"*100)


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    """Generate all visualizations."""
    print("\n" + "="*70)
    print("GENERATING DiToS MULTI-CLASS VISUALIZATIONS")
    print("="*70)
    print(f"\nOutput directory: {OUTPUT_DIR}\n")
    
    # Generate figures
    plot_original_dataset()
    plot_imbalance_selection()
    
    # DiToS pipeline for each case
    cases = calculate_case_distributions()
    for i, (case_name, case_info) in enumerate(cases.items(), 1):
        plot_ditos_pipeline_detailed(case_name, case_info, f'03_{i}_ditos_pipeline_{case_name.split()[0]}_{case_name.split()[1]}')
    
    plot_contribution_breakdown()
    plot_comprehensive_flow()
    plot_detailed_table()
    plot_before_after_comparison()
    plot_augmentation_flow_sankey()
    
    # Print summary
    print_summary()
    
    print("\n" + "="*70)
    print("ALL VISUALIZATIONS GENERATED!")
    print("="*70)
    print(f"\nFiles saved to: {OUTPUT_DIR}/")
    print("\nGenerated figures:")
    print("  01_original_dataset.png     - Original dataset distribution")
    print("  02_imbalance_selection.png  - Selected images per case")
    print("  03_*_ditos_pipeline_*.png   - DiToS stages per case")
    print("  04_contribution_breakdown.png - Stacked contribution")
    print("  05_comprehensive_flow.png   - Complete flow diagram")
    print("  06_detailed_table.png       - Detailed numbers table")
    print("  07_before_after_comparison.png - Before/After comparison")
    print("  08_augmentation_flow.png    - Waterfall flow diagram")
    print("\n  + ditos_complete_counts.csv - CSV with all numbers")


if __name__ == "__main__":
    main()
