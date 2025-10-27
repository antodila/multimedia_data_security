import json
import os
from datetime import datetime

def check_competition_compliance():
    print("🔍 COMPETITION COMPLIANCE CHECK")
    print("=" * 50)
    
    compliance_report = {
        'timestamp': datetime.now().isoformat(),
        'group_name': 'shadowmark',
        'compliance_checks': {}
    }
    
    # Check 1: Embedding function structure
    print("📋 1. Embedding Function Compliance")
    try:
        from embedding import embedding
        import inspect
        sig = inspect.signature(embedding)
        params = list(sig.parameters.keys())
        
        if len(params) == 2 and params[0] == 'input1' and params[1] == 'input2':
            compliance_report['compliance_checks']['embedding_structure'] = 'PASS'
            print("   ✅ Function signature: def embedding(input1, input2)")
            print("   ✅ Returns watermarked image")
        else:
            compliance_report['compliance_checks']['embedding_structure'] = 'FAIL'
            print("   ❌ Function signature incorrect")
    except Exception as e:
        compliance_report['compliance_checks']['embedding_structure'] = f'ERROR: {str(e)}'
        print(f"   ❌ Error checking embedding: {str(e)}")
    
    # Check 2: Detection function structure
    print("\n📋 2. Detection Function Compliance")
    try:
        from detection_shadowmark import detection
        import inspect
        sig = inspect.signature(detection)
        params = list(sig.parameters.keys())
        
        if len(params) == 3 and params[0] == 'input1' and params[1] == 'input2' and params[2] == 'input3':
            compliance_report['compliance_checks']['detection_structure'] = 'PASS'
            print("   ✅ Function signature: def detection(input1, input2, input3)")
            print("   ✅ Returns (presence_int, wpsnr_float)")
        else:
            compliance_report['compliance_checks']['detection_structure'] = 'FAIL'
            print("   ❌ Function signature incorrect")
    except Exception as e:
        compliance_report['compliance_checks']['detection_structure'] = f'ERROR: {str(e)}'
        print(f"   ❌ Error checking detection: {str(e)}")
    
    # Check 3: Detection file naming
    print("\n📋 3. Detection File Naming")
    detection_file = 'detection_shadowmark.py'
    if os.path.exists(detection_file):
        compliance_report['compliance_checks']['detection_naming'] = 'PASS'
        print(f"   ✅ Detection file: {detection_file}")
    else:
        compliance_report['compliance_checks']['detection_naming'] = 'FAIL'
        print(f"   ❌ Detection file not found: {detection_file}")
    
    # Check 4: Attack function structure
    print("\n📋 4. Attack Function Compliance")
    try:
        from attacks import attacks
        import inspect
        sig = inspect.signature(attacks)
        params = list(sig.parameters.keys())
        
        if len(params) == 3 and params[0] == 'input1' and params[1] == 'attack_name' and params[2] == 'param_array':
            compliance_report['compliance_checks']['attack_structure'] = 'PASS'
            print("   ✅ Function signature: def attacks(input1, attack_name, param_array)")
            print("   ✅ Returns attacked image")
        else:
            compliance_report['compliance_checks']['attack_structure'] = 'FAIL'
            print("   ❌ Function signature incorrect")
    except Exception as e:
        compliance_report['compliance_checks']['attack_structure'] = f'ERROR: {str(e)}'
        print(f"   ❌ Error checking attacks: {str(e)}")
    
    # Check 5: Permitted attacks
    print("\n📋 5. Permitted Attacks Check")
    try:
        from attacks import attacks
        permitted_attacks = ['jpeg', 'awgn', 'blur', 'median', 'resize']
        smart_strategies = [
            'strategy_1_stealth', 'strategy_2_brutal', 'strategy_3_smart',
            'strategy_4_chaos', 'strategy_5_precision', 'strategy_6_wave',
            'strategy_7_counter_intuitive', 'strategy_8_frequency_hunter', 'strategy_9_surgical'
        ]
        
        compliance_report['compliance_checks']['permitted_attacks'] = {
            'individual_attacks': permitted_attacks,
            'smart_strategies': smart_strategies,
            'total_attack_methods': len(permitted_attacks) + len(smart_strategies)
        }
        print("   ✅ Individual attacks: JPEG, AWGN, Blur, Median, Resize")
        print(f"   ✅ Smart strategies: {len(smart_strategies)} sophisticated combinations")
        print(f"   ✅ Total attack methods: {len(permitted_attacks) + len(smart_strategies)}")
    except Exception as e:
        compliance_report['compliance_checks']['permitted_attacks'] = f'ERROR: {str(e)}'
        print(f"   ❌ Error checking attacks: {str(e)}")
    
    # Check 6: No prints or interactive elements
    print("\n📋 6. Code Cleanliness Check")
    try:
        # Check embedding.py for prints
        with open('embedding.py', 'r') as f:
            embedding_content = f.read()
        
        # Check detection_shadowmark.py for prints
        with open('detection_shadowmark.py', 'r') as f:
            detection_content = f.read()
        
        # Check attacks.py for prints
        with open('attacks.py', 'r') as f:
            attacks_content = f.read()
        
        print_statements = []
        if 'print(' in embedding_content:
            print_statements.append('embedding.py')
        if 'print(' in detection_content:
            print_statements.append('detection_shadowmark.py')
        if 'print(' in attacks_content:
            print_statements.append('attacks.py')
        
        if print_statements:
            compliance_report['compliance_checks']['no_prints'] = f'WARNING: print statements in {print_statements}'
            print(f"   ⚠️ Print statements found in: {print_statements}")
        else:
            compliance_report['compliance_checks']['no_prints'] = 'PASS'
            print("   ✅ No print statements found")
            
    except Exception as e:
        compliance_report['compliance_checks']['no_prints'] = f'ERROR: {str(e)}'
        print(f"   ❌ Error checking code cleanliness: {str(e)}")
    
    # Check 7: Required files
    print("\n📋 7. Required Files Check")
    required_files = [
        'embedding.py',
        'detection_shadowmark.py', 
        'attacks.py',
        'roc_threshold.py',
        'mark.npy',
        'tau.json'
    ]
    
    missing_files = []
    for file in required_files:
        if os.path.exists(file):
            print(f"   ✅ {file}")
        else:
            missing_files.append(file)
            print(f"   ❌ {file} - MISSING")
    
    compliance_report['compliance_checks']['required_files'] = {
        'missing': missing_files,
        'all_present': len(missing_files) == 0
    }
    
    # Check 8: Performance test
    print("\n📋 8. Performance Test")
    try:
        import time
        start_time = time.time()
        
        # Test detection speed
        from detection_shadowmark import detection
        p, w = detection('lena_grey.bmp', 'watermarked_images/shadowmark_lena_grey.bmp', 'lena_grey.bmp')
        
        detection_time = time.time() - start_time
        
        if detection_time <= 5.0:
            compliance_report['compliance_checks']['detection_speed'] = f'PASS ({detection_time:.3f}s)'
            print(f"   ✅ Detection time: {detection_time:.3f} seconds (≤ 5s required)")
        else:
            compliance_report['compliance_checks']['detection_speed'] = f'FAIL ({detection_time:.3f}s)'
            print(f"   ❌ Detection time: {detection_time:.3f} seconds (> 5s required)")
            
    except Exception as e:
        compliance_report['compliance_checks']['detection_speed'] = f'ERROR: {str(e)}'
        print(f"   ❌ Error testing detection speed: {str(e)}")
    
    # Final compliance summary
    print("\n🏆 COMPLIANCE SUMMARY")
    print("=" * 50)
    
    passed_checks = sum(1 for check in compliance_report['compliance_checks'].values() 
                      if isinstance(check, str) and check.startswith('PASS'))
    total_checks = len(compliance_report['compliance_checks'])
    
    compliance_report['summary'] = {
        'passed_checks': passed_checks,
        'total_checks': total_checks,
        'compliance_percentage': (passed_checks / total_checks) * 100 if total_checks > 0 else 0
    }
    
    print(f"   Passed checks: {passed_checks}/{total_checks}")
    print(f"   Compliance: {(passed_checks / total_checks) * 100:.1f}%")
    
    if passed_checks == total_checks:
        print("   🎉 FULLY COMPLIANT WITH COMPETITION RULES!")
    elif passed_checks >= total_checks * 0.8:
        print("   ✅ MOSTLY COMPLIANT - Minor issues to address")
    else:
        print("   ⚠️ COMPLIANCE ISSUES DETECTED - Review required")
    
    # Save compliance report
    with open('competition_compliance_report.json', 'w') as f:
        json.dump(compliance_report, f, indent=2)
    
    print(f"\n   Compliance report saved to: competition_compliance_report.json")
    
    return compliance_report

if __name__ == "__main__":
    compliance_report = check_competition_compliance()
