#!/usr/bin/env python3
"""
Prediction Accuracy Testing Script

This script runs comprehensive accuracy tests for the flood prediction model,
testing against historical data and providing detailed performance metrics.
"""

import sys
import os
from pathlib import Path

# Add the parent directory to the path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from services.predict import run_prediction_accuracy_test
import json
from datetime import datetime

def main():
    """Run prediction accuracy testing"""
    print("🌊 Crisis Connect - Prediction Accuracy Testing")
    print("=" * 60)

    try:
        print("🔬 Starting comprehensive prediction accuracy test...")
        print("This may take several minutes depending on the test period.")

        # Run the comprehensive test
        results = run_prediction_accuracy_test()

        if 'error' in results:
            print(f"❌ Test failed: {results['error']}")
            return 1

        # Print summary results
        print("\n📊 VALIDATION RESULTS SUMMARY")
        print("-" * 40)

        validation = results['validation_results']
        assessment = results['overall_assessment']

        print(f"📅 Test Period: {validation.get('test_period_days', 'N/A')} days")
        print(f"📍 Locations Tested: {validation.get('locations_tested', 'N/A')}")
        print(f"🔢 Total Predictions: {validation.get('total_predictions', 'N/A')}")
        print(f"🌊 Actual Floods: {validation.get('total_actual_floods', 'N/A')}")
        print(f"🚨 Predicted Floods: {validation.get('total_predicted_floods', 'N/A')}")

        print(f"\n📈 ACCURACY METRICS:")
        print(f"Overall Accuracy: {validation.get('overall_accuracy', 0):.1%}")
        print(f"False Positive Rate: {validation.get('false_positive_rate', 0):.1%}")
        print(f"False Negative Rate: {validation.get('false_negative_rate', 0):.1%}")

        if 'classification_report' in validation and 'macro avg' in validation['classification_report']:
            macro = validation['classification_report']['macro avg']
            print(f"Precision (avg): {macro.get('precision', 0):.3f}")
            print(f"Recall (avg): {macro.get('recall', 0):.3f}")
            print(f"F1-Score (avg): {macro.get('f1-score', 0):.3f}")

        print(f"\n🎯 RISK DISTRIBUTION:")
        risk_dist = validation.get('risk_distribution', {})
        print(f"High Risk (>70): {risk_dist.get('high_risk_percentage', 0):.1f}%")
        print(f"Medium Risk (40-70): {risk_dist.get('medium_risk_percentage', 0):.1f}%")
        print(f"Low Risk (<40): {risk_dist.get('low_risk_percentage', 0):.1f}%")

        print(f"\n🏆 OVERALL ASSESSMENT: {assessment.get('assessment', 'unknown').upper()}")
        print(f"Confidence Level: {assessment.get('confidence_level', 'unknown')}")

        print(f"\n💡 RECOMMENDATIONS:")
        for rec in assessment.get('recommendations', ['No recommendations available']):
            print(f"• {rec}")

        print(f"\n📄 Detailed report saved to: prediction_accuracy_report.json")

        # Print stress test results
        print(f"\n🧪 STRESS TEST RESULTS:")
        for test in results.get('stress_test_results', []):
            status = "✅" if 'error' not in test else "❌"
            risk = f"{test.get('risk_score', 'N/A'):.1f}" if 'risk_score' in test else "ERROR"
            print(f"{status} {test.get('scenario', 'Unknown')}: Risk Score {risk}")

        # Print edge case results
        print(f"\n🔍 EDGE CASE TEST RESULTS:")
        for test in results.get('edge_case_results', []):
            status = "✅" if test.get('handled_gracefully', False) else "❌"
            risk = f"{test.get('risk_score', 'N/A'):.1f}" if 'risk_score' in test else "ERROR"
            print(f"{status} {test.get('case', 'Unknown')}: Risk Score {risk}")

        print("\n✅ Prediction accuracy testing completed successfully!")
        return 0

    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())
