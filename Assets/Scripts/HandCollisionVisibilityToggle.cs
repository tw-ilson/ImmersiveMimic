using UnityEngine;
using Oculus.Interaction.Input;

public class HandCollisionVisibilityToggle : MonoBehaviour
{
    [SerializeField]
    private MeshRenderer visualSphereMeshRenderer;

    private void Start()
    {
        // If not assigned in inspector, try to find VisualSphere as child
        if (visualSphereMeshRenderer == null)
        {
            Transform visualSphere = transform.Find("VisualSphere");
            if (visualSphere != null)
            {
                visualSphereMeshRenderer = visualSphere.GetComponent<MeshRenderer>();
            }
        }

        // Start with sphere invisible
        if (visualSphereMeshRenderer != null)
        {
            visualSphereMeshRenderer.enabled = false;
        }
        
    }

    private void OnTriggerEnter(Collider other)
    {
        // Check if the colliding object is part of the OVRHand system
        if (IsHandTrackingObject(other))
        {
            Debug.Log("Collider trigered:"+other.gameObject.name);
            if (visualSphereMeshRenderer != null)
            {
                visualSphereMeshRenderer.enabled = true;
            }
        }
    }

    private void OnTriggerExit(Collider other)
    {
        // Check if the colliding object is part of the OVRHand system
        if (IsHandTrackingObject(other))
        {
            if (visualSphereMeshRenderer != null)
            {
                visualSphereMeshRenderer.enabled = false;
            }
        }
    }

    private bool IsHandTrackingObject(Collider other)
    {
        // Check for various OVR hand-related components
        return other.GetComponent<OVRHand>() != null ||
               other.GetComponentInParent<OVRHand>() != null ||
               // other.gameObject.layer == LayerMask.NameToLayer("HandsLayer") ||
               other.CompareTag("Hand");
    }
}