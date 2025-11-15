import json
from datetime import datetime
def save_attention_data(data, filename):
    try:
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        return True
    except Exception as e:
        print(f"Error saving attention data: {e}")
        return False

def load_attention_data(filename):
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except:
        return []

class AttentionAnalyzer:
    def __init__(self):
        self.session_data = []
        
    def load_historical_data(self, filename):
        self.session_data = load_attention_data(filename)
        return self.session_data
    
    def calculate_attention_patterns(self):
        if not self.session_data:
            return None
            
        focus_percentages = [session.get('focus_percentage', 0) for session in self.session_data]
        session_durations = [session.get('session_time', 0) for session in self.session_data]
        distraction_counts = [session.get('distraction_count', 0) for session in self.session_data]
        
        patterns = {
            'avg_focus': sum(focus_percentages) / len(focus_percentages) if focus_percentages else 0,
            'avg_duration': sum(session_durations) / len(session_durations) if session_durations else 0,
            'avg_distractions': sum(distraction_counts) / len(distraction_counts) if distraction_counts else 0,
            'total_sessions': len(self.session_data),
            'best_session_focus': max(focus_percentages) if focus_percentages else 0,
            'worst_session_focus': min(focus_percentages) if focus_percentages else 0
        }
        
        return patterns
    
    def generate_suggestions(self, current_session_data=None):
        patterns = self.calculate_attention_patterns()
        suggestions = []
        if patterns['avg_focus'] < 60:
            suggestions.append({
                'type': 'focus_improvement',
                'title': 'Focus Improvement Needed',
                'message': f'Your average focus is {patterns["avg_focus"]:.1f}%.',
            })
        elif patterns['avg_focus'] < 80:
            suggestions.append({
                'type': 'focus_ok',
                'title': 'Good Focus, Can Improve',
                'message': f'Your average focus is {patterns["avg_focus"]:.1f}%.',
            })
        else:
            suggestions.append({
                'type': 'focus_excellence',
                'title': 'Excellent Focus!',
                'message': f'Your average focus is {patterns["avg_focus"]:.1f}%.',
            })
        
        avg_duration_minutes = patterns['avg_duration'] / 60
        if avg_duration_minutes > 60:
            suggestions.append({
                'type': 'duration_warning',
                'title': 'Session Duration Warning',
                'message': f'Your sessions are quite long ({avg_duration_minutes:.1f} min). Consider 25-45 min focused sessions with breaks.',
            })
        elif avg_duration_minutes < 15:
            suggestions.append({
                'type': 'duration_improvement',
                'title': 'Build Longer Sessions',
                'message': f'Try extending your sessions to 25-45 minutes for better focus development.',
            })
        if patterns['avg_distractions'] > 5:
            suggestions.append({
                'type': 'distraction_reduction',
                'title': 'High Distraction Count',
                'message': f'You average {patterns["avg_distractions"]:.1f} distractions per session. Try the Pomodoro Technique.',
            })
        
        return suggestions
    
    
    def calculate_optimal_session_duration(self):

        patterns = self.calculate_attention_patterns()
        if not patterns:
            return 1500 
        
        if patterns['avg_focus'] > 80:
            return min(3600, max(1800, int(patterns['avg_duration'] * 1.2)))  
        elif patterns['avg_focus'] > 60:
            return 1800  
        else:
            return max(900, int(patterns['avg_duration'] * 0.8))  
    
    def get_focus_improvement_recommendations(self):
        patterns = self.calculate_attention_patterns()
        
        recommendations = []
        
        if patterns['avg_duration'] > 1800: 
            recommendations.append("Try the Pomodoro Technique: 25 min focus + 5 min break")
        
        if patterns['avg_distractions'] > 3:
            recommendations.extend([
                "Put your phone in another room or use focus mode",
                "Create a dedicated, distraction-free workspace"
            ])
        
        if patterns['avg_focus'] < 70:
            recommendations.extend([
                "Start with shorter sessions (15-20 min) and gradually increase",
                "Take regular breaks every 25-30 minutes",
                "Set specific, achievable goals for each session"
            ])
        elif patterns['avg_focus'] < 85:
            recommendations.extend([
                "Maintain consistent session times to build routine",
                "Track what activities improve your focus",
                "Consider meditation or mindfulness exercises"
            ])
        
        return recommendations

        
    def generate_session_report(self, current_session_data):
        patterns = self.calculate_attention_patterns()
        
        report = {
            'session_summary': {
                'duration': current_session_data.get('session_time', 0),
                'focus_time': current_session_data.get('focus_time', 0),
                'focus_percentage': current_session_data.get('focus_percentage', 0),
                'distractions': current_session_data.get('distraction_count', 0),
                'max_attention_span': current_session_data.get('max_attention_span', 0)
            },
            'historical_comparison': {},
            'suggestions': self.generate_suggestions(),
            'recommended_next_session': self.calculate_optimal_session_duration(),
            'improvement_recommendations': self.get_focus_improvement_recommendations()
        }
        
        if patterns:
            report['historical_comparison'] = {
                'vs_average_focus': current_session_data.get('focus_percentage', 0) - patterns['avg_focus'],
                'vs_average_duration': current_session_data.get('session_time', 0) - patterns['avg_duration'],
                'vs_average_distractions': current_session_data.get('distraction_count', 0) - patterns['avg_distractions']
            }
        
        return report

def display_suggestions(suggestions):
    if not suggestions:
        print("No suggestions available.")
        return
    
    for suggestion in suggestions:
        print(f"\n• {suggestion['title']}")
        print(f"  {suggestion['message']}")
    
    print("="*50)

def save_session_report(report, filename=None):
    """Save session report to file"""
    if filename is None:
        report_filename = f"session_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    else:
        report_filename = filename
        
    try:
        with open(report_filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        print(f"Session report saved to {report_filename}")
        return report_filename
    except Exception as e:
        print(f"Error saving report: {e}")
        return None
