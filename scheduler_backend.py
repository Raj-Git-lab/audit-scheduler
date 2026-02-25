import pandas as pd
import numpy as np
from collections import defaultdict
import time
from datetime import timedelta
from io import BytesIO

# Define scheduling patterns
SCHEDULING_PATTERNS = {
    'WEEKLY': list(range(1, 53)),
    'BIWEEKLY': [1] + list(range(3, 53, 2)),
    'MONTHLY': [1, 5, 9, 13, 18, 22, 26, 31, 35, 39, 44, 48],
    'QUARTERLY': [1, 13, 26, 39],
    'HALF_YEARLY': [1, 26],
    'YEARLY': [1]
}


def get_scheduling_pattern(min_audits):
    """Determine the scheduling pattern based on minimum audit count"""
    if min_audits >= 52:
        return SCHEDULING_PATTERNS['WEEKLY']
    elif min_audits >= 26:
        return SCHEDULING_PATTERNS['BIWEEKLY']
    elif min_audits >= 12:
        return SCHEDULING_PATTERNS['MONTHLY']
    elif min_audits >= 4:
        return SCHEDULING_PATTERNS['QUARTERLY']
    elif min_audits >= 2:
        return SCHEDULING_PATTERNS['HALF_YEARLY']
    else:
        return SCHEDULING_PATTERNS['YEARLY']


def get_priority_score(row, log_callback=None):
    """Calculate priority score based on risk score and percentile group"""
    try:
        risk_score = float(str(row['risk_score']))
        percentile_group = str(row['percentile_group'])

        priority = risk_score * 10000

        # RS 10.2 (escalated) gets HIGHEST priority
        if abs(risk_score - 10.2) < 0.0001:
            priority += 5000
            if log_callback:
                log_callback(f"Found RS 10.2 class: {row['class_name']}, Risk Score: {risk_score}")
            return priority

        # Add percentile priority for non-10.2 classes
        if '80th Percentile' in percentile_group:
            priority += 3000
        elif '50th Percentile' in percentile_group:
            priority += 2000
        elif '20th Percentile' in percentile_group:
            priority += 1000
        elif '0 FN' in percentile_group:
            if abs(risk_score - 10.0) < 0.0001:
                priority += 500
            else:
                priority = -1

        return priority
    except Exception as e:
        if log_callback:
            log_callback(f"Error calculating priority score: {str(e)}")
        return -1


def calculate_node_capacity(hr_metrics, log_callback=None):
    """Calculate yearly and weekly capacity for each node"""
    yearly_capacity_dict = {}
    weekly_capacity_dict = {}

    HOURS_PER_WEEK = 40
    WEEKS_PER_YEAR = 52

    capacity_details = []

    try:
        for _, row in hr_metrics.iterrows():
            node = row['Node']

            if pd.isna(node):
                continue

            hc = float(row['HC']) if pd.notna(row['HC']) else 0
            aht_minutes = float(row['AHT']) if pd.notna(row['AHT']) else 60
            shrinkage = float(row['Shrinkage']) if pd.notna(row['Shrinkage']) else 0

            escalation = float(row['Escalation']) if 'Escalation' in row and pd.notna(
                row.get('Escalation')) else 1
            new_launch = float(row['new_launch']) if 'new_launch' in row and pd.notna(
                row.get('new_launch')) else 1
            image_review = float(row['image_review']) if 'image_review' in row and pd.notna(
                row.get('image_review')) else 1

            aht_hours = aht_minutes / 60

            if shrinkage > 1:
                shrinkage = shrinkage / 100

            productive_hours_per_week = HOURS_PER_WEEK * shrinkage * escalation * new_launch * image_review

            if aht_hours > 0:
                audits_per_auditor_per_week = productive_hours_per_week / aht_hours
            else:
                audits_per_auditor_per_week = 0

            weekly_capacity = int(hc * audits_per_auditor_per_week)
            yearly_capacity = weekly_capacity * WEEKS_PER_YEAR

            weekly_capacity_dict[node] = weekly_capacity
            yearly_capacity_dict[node] = yearly_capacity

            capacity_details.append({
                'Node': node,
                'HC': hc,
                'AHT (min)': aht_minutes,
                'Shrinkage': shrinkage,
                'Weekly Capacity': weekly_capacity,
                'Yearly Capacity': yearly_capacity
            })

        if len(yearly_capacity_dict) == 0:
            raise ValueError("No valid nodes found in HR metrics")

        return yearly_capacity_dict, weekly_capacity_dict, pd.DataFrame(capacity_details)

    except Exception as e:
        raise Exception(f"Error calculating node capacity: {str(e)}")


def get_required_gap(min_audits):
    """Get the required gap between audits"""
    if min_audits >= 52:
        return 1
    elif min_audits >= 24:
        return 2
    elif min_audits >= 12:
        return 4
    elif min_audits >= 6:
        return 8
    elif min_audits >= 4:
        return 12
    elif min_audits >= 2:
        return 26
    else:
        return 52


def circular_gap(w1, w2, total_weeks=52):
    """
    ✅ FIX #1: Proper circular distance between two weeks.
    e.g., week 50 and week 3 are 5 apart (not 47).
    """
    diff = abs(w1 - w2)
    return min(diff, total_weeks - diff)


def prepare_data(classes_data, hr_metrics, log_callback=None):
    """Prepare and validate data"""

    required_columns = ['node', 'class_name', 'risk_score', 'percentile_group', 'minimum_audit_count']
    missing_columns = [col for col in required_columns if col not in classes_data.columns]

    if missing_columns:
        raise ValueError(f"Missing required columns in audit file: {missing_columns}")

    classes_data = classes_data.copy()
    classes_data['batch_marketplace'] = classes_data['node']
    classes_data['class_key'] = classes_data['class_name']

    classes_data['risk_score'] = pd.to_numeric(classes_data['risk_score'], errors='coerce').fillna(5.0)
    classes_data['minimum_audit_count'] = pd.to_numeric(
        classes_data['minimum_audit_count'], errors='coerce'
    ).fillna(1).astype(int)
    classes_data.loc[classes_data['minimum_audit_count'] < 0, 'minimum_audit_count'] = 1

    classes_data['priority_score'] = classes_data.apply(
        lambda row: get_priority_score(row, log_callback), axis=1
    )

    if classes_data['class_key'].duplicated().any():
        classes_data['class_key'] = classes_data['class_name'] + '_' + classes_data.index.astype(str)

    yearly_capacity_dict, weekly_capacity_dict, capacity_df = calculate_node_capacity(hr_metrics, log_callback)

    total_demand = defaultdict(int)
    priority_demand = defaultdict(int)
    backlog_demand = defaultdict(int)

    for _, row in classes_data.iterrows():
        node = row['batch_marketplace']
        min_audits = int(row['minimum_audit_count'])

        if pd.notna(node):
            total_demand[node] += min_audits
            if row['priority_score'] >= 0:
                priority_demand[node] += min_audits
            else:
                backlog_demand[node] += min_audits

    return (classes_data, yearly_capacity_dict, weekly_capacity_dict,
            total_demand, priority_demand, backlog_demand, capacity_df)


def distribute_audits(classes_data, yearly_capacity_dict, weekly_capacity_dict,
                      total_demand, priority_demand, backlog_demand, progress_callback=None):
    """Distribute audits with gap-based scheduling — FIXED VERSION"""

    weeks = list(range(1, 53))
    audit_schedule = pd.DataFrame(
        0,
        index=classes_data['class_key'].unique(),
        columns=weeks,
        dtype='int8'
    )
    weekly_loads = defaultdict(lambda: defaultdict(int))
    scheduling_results = defaultdict(list)

    # =====================================================
    # ✅ FIX #2: Completely rewritten week-finding function
    # =====================================================
    def find_weeks_with_gap(node, required_weeks, required_gap, weekly_capacity, start_week=1):
        """
        Find weeks respecting gap and capacity.
        Uses circular gap so wrap-around works correctly.
        Tries multiple starting points if first attempt fails.
        """
        best_selected = []
        best_within_capacity = False

        # Try different starting weeks to maximize chances
        start_weeks_to_try = list(range(1, 53))

        for try_start in start_weeks_to_try:
            selected_weeks = []
            current_week = try_start

            for _ in range(required_weeks):
                found = False
                for offset in range(52):
                    check_week = ((current_week + offset - 1) % 52) + 1

                    # Check capacity
                    if weekly_loads[node][check_week] < weekly_capacity:
                        # ✅ FIX: Use circular_gap instead of abs()
                        if not selected_weeks or all(
                            circular_gap(check_week, w) >= required_gap
                            for w in selected_weeks
                        ):
                            selected_weeks.append(check_week)
                            current_week = ((check_week + required_gap - 1) % 52) + 1
                            found = True
                            break

                if not found:
                    break

            if len(selected_weeks) == required_weeks:
                return sorted(selected_weeks), True

            # Keep track of best partial result
            if len(selected_weeks) > len(best_selected):
                best_selected = selected_weeks

            # If we found at least something from first start, don't try all 52
            if len(best_selected) >= required_weeks - 1 and try_start > 5:
                break

        # =====================================================
        # ✅ FIX #3: Better fallback — try with relaxed gap
        # =====================================================
        # Try with progressively smaller gaps
        for reduced_gap in range(max(1, required_gap - 1), 0, -1):
            selected_weeks = []
            current_week = 1

            for _ in range(required_weeks):
                found = False
                for offset in range(52):
                    check_week = ((current_week + offset - 1) % 52) + 1

                    if weekly_loads[node][check_week] < weekly_capacity:
                        if not selected_weeks or all(
                            circular_gap(check_week, w) >= reduced_gap
                            for w in selected_weeks
                        ):
                            selected_weeks.append(check_week)
                            current_week = ((check_week + reduced_gap - 1) % 52) + 1
                            found = True
                            break

                if not found:
                    break

            if len(selected_weeks) == required_weeks:
                return sorted(selected_weeks), True

        # =====================================================
        # ✅ FIX #4: Final fallback — evenly spaced, ignore capacity
        # =====================================================
        selected_weeks = []
        ideal_gap = max(1, 52 // required_weeks)
        current_week = 1

        for i in range(required_weeks):
            week = ((current_week + i * ideal_gap - 1) % 52) + 1
            selected_weeks.append(week)

        return sorted(set(selected_weeks))[:required_weeks], False

    # Track statistics
    stats = {
        'skipped_no_node': 0,
        'skipped_zero_audits': 0,
        'skipped_no_capacity': 0,
        'within_capacity': 0,
        'over_capacity': 0,
        'zfn_scheduled': 0,
        'zfn_not_scheduled': 0
    }

    classes_to_skip = set()

    # First pass: identify invalid classes
    for _, row in classes_data.iterrows():
        class_key = row['class_key']
        class_name = row['class_name']
        node = row['batch_marketplace']
        min_audits = int(row['minimum_audit_count']) if pd.notna(row['minimum_audit_count']) else 0
        risk_score = row['risk_score']
        priority_score = row['priority_score']

        if pd.isna(node) or node == '':
            scheduling_results['NO_NODE'].append({
                'class_key': class_key,
                'class_name': class_name,
                'status': 'No Node Assigned',
                'scheduled_weeks': [],
                'min_audits': min_audits,
                'required_gap': 0,
                'priority_score': priority_score,
                'risk_score': risk_score
            })
            classes_to_skip.add(class_key)
            stats['skipped_no_node'] += 1
            continue

        if min_audits <= 0:
            scheduling_results[node].append({
                'class_key': class_key,
                'class_name': class_name,
                'status': 'Zero Audits Required',
                'scheduled_weeks': [],
                'min_audits': min_audits,
                'required_gap': 0,
                'priority_score': priority_score,
                'risk_score': risk_score
            })
            classes_to_skip.add(class_key)
            stats['skipped_zero_audits'] += 1

    # Process each node
    unique_nodes = [n for n in classes_data['batch_marketplace'].unique() if pd.notna(n)]
    total_nodes = len(unique_nodes)

    for node_idx, node in enumerate(sorted(unique_nodes)):
        if progress_callback:
            progress_callback((node_idx + 1) / total_nodes, f"Processing node: {node}")

        weekly_capacity = weekly_capacity_dict.get(node, 0)
        yearly_capacity = yearly_capacity_dict.get(node, 0)

        for week in weeks:
            weekly_loads[node][week] = 0

        if weekly_capacity <= 0:
            node_classes = classes_data[classes_data['batch_marketplace'] == node].copy()
            for _, row in node_classes.iterrows():
                class_key = row['class_key']
                if class_key in classes_to_skip:
                    continue

                scheduling_results[node].append({
                    'class_key': class_key,
                    'class_name': row['class_name'],
                    'status': 'No Capacity',
                    'scheduled_weeks': [],
                    'min_audits': int(row['minimum_audit_count']),
                    'required_gap': 0,
                    'priority_score': row['priority_score'],
                    'risk_score': row['risk_score']
                })
                stats['skipped_no_capacity'] += 1
            continue

        node_classes = classes_data[classes_data['batch_marketplace'] == node].copy()
        node_classes = node_classes[~node_classes['class_key'].isin(classes_to_skip)]

        priority_classes = node_classes[node_classes['priority_score'] >= 0].copy()
        zfn_classes = node_classes[node_classes['priority_score'] < 0].copy()

        priority_classes = priority_classes.sort_values(
            ['priority_score', 'risk_score', 'minimum_audit_count'],
            ascending=[False, False, False]
        )

        # =====================================================
        # Phase 1: Schedule Priority Classes (NON-ZFN)
        # =====================================================
        for idx, row in priority_classes.iterrows():
            class_key = row['class_key']
            class_name = row['class_name']
            min_audits = int(row['minimum_audit_count'])
            risk_score = row['risk_score']
            priority_score = row['priority_score']

            required_gap = get_required_gap(min_audits)

            selected_weeks, within_capacity = find_weeks_with_gap(
                node, min_audits, required_gap, weekly_capacity
            )

            # ✅ FIX #5: Ensure we always have enough weeks
            # If selected_weeks is shorter than min_audits, fill remaining
            if len(selected_weeks) < min_audits:
                existing = set(selected_weeks)
                # Add least-loaded weeks that aren't already selected
                remaining_weeks = sorted(
                    [w for w in weeks if w not in existing],
                    key=lambda w: weekly_loads[node][w]
                )
                for w in remaining_weeks:
                    if len(selected_weeks) >= min_audits:
                        break
                    selected_weeks.append(w)
                selected_weeks = sorted(selected_weeks)
                within_capacity = False

            if within_capacity and len(selected_weeks) == min_audits:
                for week in selected_weeks:
                    audit_schedule.loc[class_key, week] = 1
                    weekly_loads[node][week] += 1

                scheduling_results[node].append({
                    'class_key': class_key,
                    'class_name': class_name,
                    'status': 'Within Capacity',
                    'scheduled_weeks': selected_weeks,
                    'min_audits': min_audits,
                    'required_gap': required_gap,
                    'priority_score': priority_score,
                    'risk_score': risk_score
                })
                stats['within_capacity'] += 1
            else:
                for week in selected_weeks[:min_audits]:
                    audit_schedule.loc[class_key, week] = 2
                    weekly_loads[node][week] += 1

                scheduling_results[node].append({
                    'class_key': class_key,
                    'class_name': class_name,
                    'status': 'Over Capacity',
                    'scheduled_weeks': selected_weeks[:min_audits],
                    'min_audits': min_audits,
                    'required_gap': required_gap,
                    'priority_score': priority_score,
                    'risk_score': risk_score
                })
                stats['over_capacity'] += 1

        # =====================================================
        # Phase 2: Fill with ZFN Classes — FIXED
        # =====================================================
        remaining_capacity = {}
        for week in weeks:
            remaining_capacity[week] = weekly_capacity - weekly_loads[node][week]

        total_remaining = sum(max(0, cap) for cap in remaining_capacity.values())

        if len(zfn_classes) > 0:
            zfn_classes = zfn_classes.sort_values(
                ['risk_score', 'minimum_audit_count'],
                ascending=[False, False]
            )

            for idx, row in zfn_classes.iterrows():
                class_key = row['class_key']
                class_name = row['class_name']
                min_audits = int(row['minimum_audit_count'])
                risk_score = row['risk_score']
                priority_score = row['priority_score']

                required_gap = get_required_gap(min_audits)

                # =====================================================
                # ✅ FIX #6: Multi-pass ZFN scheduling with gap relaxation
                # =====================================================
                selected_weeks = []
                scheduled = False

                # Pass 1: Try with full gap and remaining capacity
                for try_gap in range(required_gap, 0, -1):
                    selected_weeks = []
                    current_week = 1

                    for attempt in range(min_audits):
                        found = False
                        for offset in range(52):
                            check_week = ((current_week + offset - 1) % 52) + 1

                            if remaining_capacity.get(check_week, 0) > 0:
                                # ✅ FIX: Use circular_gap
                                if not selected_weeks or all(
                                    circular_gap(check_week, w) >= try_gap
                                    for w in selected_weeks
                                ):
                                    selected_weeks.append(check_week)
                                    current_week = ((check_week + try_gap - 1) % 52) + 1
                                    found = True
                                    break

                        if not found:
                            break

                    if len(selected_weeks) == min_audits:
                        scheduled = True
                        break

                if scheduled:
                    for week in selected_weeks:
                        audit_schedule.loc[class_key, week] = 1
                        weekly_loads[node][week] += 1
                        remaining_capacity[week] -= 1

                    scheduling_results[node].append({
                        'class_key': class_key,
                        'class_name': class_name,
                        'status': 'ZFN',
                        'scheduled_weeks': sorted(selected_weeks),
                        'min_audits': min_audits,
                        'required_gap': required_gap,
                        'priority_score': priority_score,
                        'risk_score': risk_score
                    })
                    stats['zfn_scheduled'] += 1
                else:
                    # ✅ FIX #7: Even ZFN gets a fallback — schedule in least-loaded weeks
                    selected_weeks = []
                    available_weeks = sorted(
                        weeks,
                        key=lambda w: remaining_capacity.get(w, 0),
                        reverse=True
                    )

                    for w in available_weeks:
                        if len(selected_weeks) >= min_audits:
                            break
                        if remaining_capacity.get(w, 0) > 0:
                            if not selected_weeks or all(
                                circular_gap(w, sw) >= max(1, required_gap // 2)
                                for sw in selected_weeks
                            ):
                                selected_weeks.append(w)

                    # If still not enough, add any available weeks
                    if len(selected_weeks) < min_audits:
                        for w in available_weeks:
                            if len(selected_weeks) >= min_audits:
                                break
                            if w not in selected_weeks and remaining_capacity.get(w, 0) > 0:
                                selected_weeks.append(w)

                    if len(selected_weeks) > 0:
                        for week in selected_weeks:
                            audit_schedule.loc[class_key, week] = 1
                            weekly_loads[node][week] += 1
                            remaining_capacity[week] -= 1

                        scheduling_results[node].append({
                            'class_key': class_key,
                            'class_name': class_name,
                            'status': 'ZFN - Partial' if len(selected_weeks) < min_audits else 'ZFN',
                            'scheduled_weeks': sorted(selected_weeks),
                            'min_audits': min_audits,
                            'required_gap': required_gap,
                            'priority_score': priority_score,
                            'risk_score': risk_score
                        })
                        if len(selected_weeks) == min_audits:
                            stats['zfn_scheduled'] += 1
                        else:
                            stats['zfn_not_scheduled'] += 1
                    else:
                        scheduling_results[node].append({
                            'class_key': class_key,
                            'class_name': class_name,
                            'status': 'ZFN - Not Scheduled',
                            'scheduled_weeks': [],
                            'min_audits': min_audits,
                            'required_gap': required_gap,
                            'priority_score': priority_score,
                            'risk_score': risk_score
                        })
                        stats['zfn_not_scheduled'] += 1

    return audit_schedule, scheduling_results, weekly_loads, stats


def create_excel_output(audit_schedule, scheduling_results,
                        yearly_capacity_dict, weekly_capacity_dict,
                        total_demand, priority_demand, backlog_demand,
                        classes_data, weekly_loads):
    """Create Excel output and return as BytesIO object"""

    output = BytesIO()

    class_lookup = {}
    for _, row in classes_data.iterrows():
        class_key = row['class_key']
        class_lookup[class_key] = {
            'class_name': row['class_name'],
            'marketplace': row.get('marketplace', ''),
            'node': row['batch_marketplace'],
            'risk_score': row['risk_score'],
            'priority_score': row['priority_score'],
            'percentile_group': row['percentile_group'],
            'minimum_audit_count': row['minimum_audit_count']
        }

    status_lookup = {}
    for node in scheduling_results:
        for result in scheduling_results[node]:
            status_lookup[result['class_key']] = result['status']

    # Sheet 1: Audit Schedule
    schedule_data = []
    for class_key in audit_schedule.index:
        class_info = class_lookup.get(class_key, {})
        row_data = {
            'Class Name': class_info.get('class_name', class_key),
            'Marketplace': class_info.get('marketplace', ''),
            'Node': class_info.get('node', ''),
            'Risk Score': class_info.get('risk_score', 0),
            'Percentile Group': class_info.get('percentile_group', ''),
            'Min Audits': class_info.get('minimum_audit_count', 0),
            'Priority Score': class_info.get('priority_score', 0),
            'Schedule Status': status_lookup.get(class_key, 'Unknown')
        }

        for week in range(1, 53):
            row_data[f'Week {week}'] = audit_schedule.loc[class_key, week]

        schedule_data.append(row_data)

    schedule_output = pd.DataFrame(schedule_data)

    # Sheet 2: Summary by Node
    summary_data = []
    for node in sorted(weekly_loads.keys()):
        node_results = scheduling_results.get(node, [])
        weekly_capacity = weekly_capacity_dict.get(node, 0)
        yearly_capacity = yearly_capacity_dict.get(node, 0)

        within_capacity = sum(1 for r in node_results if r['status'] == 'Within Capacity')
        over_capacity = sum(1 for r in node_results if r['status'] == 'Over Capacity')
        zfn_count = sum(1 for r in node_results if r['status'] in ['ZFN', 'ZFN - Partial'])

        loads = list(weekly_loads[node].values())
        max_load = max(loads) if loads else 0
        avg_load = sum(loads) / 52 if loads else 0
        total_scheduled = sum(loads) if loads else 0

        over_capacity_weeks = [w for w, load in weekly_loads[node].items() if load > weekly_capacity]

        summary_data.append({
            'Node': node,
            'Weekly Capacity': weekly_capacity,
            'Yearly Capacity': yearly_capacity,
            'Total Classes': len(node_results),
            'Within Capacity': within_capacity,
            'Over Capacity': over_capacity,
            'ZFN Classes': zfn_count,
            'Max Weekly Load': max_load,
            'Avg Weekly Load': round(avg_load, 1),
            'Total Scheduled': total_scheduled,
            'Total Demand': total_demand.get(node, 0),
            'Priority Demand': priority_demand.get(node, 0),
            'Backlog Demand': backlog_demand.get(node, 0),
            'Utilization %': round((total_scheduled / yearly_capacity * 100), 1) if yearly_capacity > 0 else 0,
            'Over Capacity Weeks': len(over_capacity_weeks)
        })

    summary_df = pd.DataFrame(summary_data)

    # Sheet 3: Detailed Results
    detailed_data = []
    for node in sorted(scheduling_results.keys()):
        for result in scheduling_results[node]:
            class_key = result['class_key']
            class_info = class_lookup.get(class_key, {})

            detailed_data.append({
                'Node': node,
                'Class Name': class_info.get('class_name', class_key),
                'Risk Score': class_info.get('risk_score', 0),
                'Percentile Group': class_info.get('percentile_group', ''),
                'Min Audits': class_info.get('minimum_audit_count', 0),
                'Status': result['status'],
                'Priority Score': result['priority_score'],
                'Required Gap (weeks)': result.get('required_gap', ''),
                'Scheduled Weeks': ', '.join(map(str, result['scheduled_weeks'])),
                'Number of Weeks': len(result['scheduled_weeks'])
            })

    detailed_df = pd.DataFrame(detailed_data)

    # Sheet 4: Weekly Load
    weekly_load_data = []
    for node in sorted(weekly_loads.keys()):
        capacity = weekly_capacity_dict.get(node, 0)
        for week in range(1, 53):
            load = weekly_loads[node][week]
            weekly_load_data.append({
                'Node': node,
                'Week': week,
                'Load': load,
                'Capacity': capacity,
                'Available': capacity - load,
                'Utilization %': round((load / capacity * 100), 1) if capacity > 0 else 0,
                'Over Capacity': 'Yes' if load > capacity else 'No'
            })

    weekly_load_df = pd.DataFrame(weekly_load_data)

    # Sheet 5: Demand vs Capacity
    demand_capacity_data = []
    for node in sorted(total_demand.keys()):
        total_node_demand = total_demand.get(node, 0)
        priority_node_demand = priority_demand.get(node, 0)
        backlog_node_demand = backlog_demand.get(node, 0)
        yearly_capacity = yearly_capacity_dict.get(node, 0)
        weekly_capacity = weekly_capacity_dict.get(node, 0)

        demand_capacity_data.append({
            'Node': node,
            'Total Demand': total_node_demand,
            'Priority Demand': priority_node_demand,
            'Backlog Demand': backlog_node_demand,
            'Yearly Capacity': yearly_capacity,
            'Weekly Capacity': weekly_capacity,
            'Surplus/Deficit': yearly_capacity - total_node_demand,
            'Demand/Capacity Ratio %': round((total_node_demand / yearly_capacity * 100),
                                             1) if yearly_capacity > 0 else 0,
            'Status': 'Over Capacity' if total_node_demand > yearly_capacity else 'Within Capacity'
        })

    demand_capacity_df = pd.DataFrame(demand_capacity_data)

    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        workbook = writer.book

        header_format = workbook.add_format({
            'bold': True,
            'bg_color': '#4472C4',
            'font_color': 'white',
            'border': 1,
            'align': 'center',
            'valign': 'vcenter'
        })

        within_capacity_format = workbook.add_format({
            'bg_color': '#C6EFCE',
            'font_color': '#006100',
            'border': 1,
            'align': 'center'
        })

        over_capacity_format = workbook.add_format({
            'bg_color': '#FFC7CE',
            'font_color': '#9C0006',
            'border': 1,
            'align': 'center'
        })

        zfn_format = workbook.add_format({
            'bg_color': '#FFEB9C',
            'font_color': '#9C6500',
            'border': 1,
            'align': 'center'
        })

        schedule_output.to_excel(writer, sheet_name='Audit Schedule', index=False)
        worksheet = writer.sheets['Audit Schedule']
        for col_num, value in enumerate(schedule_output.columns.values):
            worksheet.write(0, col_num, value, header_format)
        worksheet.set_column('A:A', 40)
        worksheet.set_column('B:H', 15)
        worksheet.freeze_panes(1, 8)

        status_col = 7
        first_week_col = 8
        for row_idx in range(len(schedule_output)):
            excel_row = row_idx + 1
            status = schedule_output.iloc[row_idx]['Schedule Status']
            if status == 'Within Capacity':
                worksheet.write(excel_row, status_col, status, within_capacity_format)
            elif status == 'Over Capacity':
                worksheet.write(excel_row, status_col, status, over_capacity_format)
            elif 'ZFN' in status:
                worksheet.write(excel_row, status_col, status, zfn_format)

            for week_offset in range(52):
                col_idx = first_week_col + week_offset
                cell_value = schedule_output.iloc[row_idx, col_idx]
                if cell_value == 1:
                    worksheet.write(excel_row, col_idx, 'Y', within_capacity_format)
                elif cell_value == 2:
                    worksheet.write(excel_row, col_idx, 'N', over_capacity_format)

        summary_df.to_excel(writer, sheet_name='Summary by Node', index=False)
        worksheet = writer.sheets['Summary by Node']
        for col_num, value in enumerate(summary_df.columns.values):
            worksheet.write(0, col_num, value, header_format)

        detailed_df.to_excel(writer, sheet_name='Detailed Results', index=False)
        worksheet = writer.sheets['Detailed Results']
        for col_num, value in enumerate(detailed_df.columns.values):
            worksheet.write(0, col_num, value, header_format)

        weekly_load_df.to_excel(writer, sheet_name='Weekly Load', index=False)
        worksheet = writer.sheets['Weekly Load']
        for col_num, value in enumerate(weekly_load_df.columns.values):
            worksheet.write(0, col_num, value, header_format)

        demand_capacity_df.to_excel(writer, sheet_name='Demand vs Capacity', index=False)
        worksheet = writer.sheets['Demand vs Capacity']
        for col_num, value in enumerate(demand_capacity_df.columns.values):
            worksheet.write(0, col_num, value, header_format)

    output.seek(0)
    return output, summary_df, detailed_df, demand_capacity_df